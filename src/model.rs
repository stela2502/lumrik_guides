use std::collections::HashMap;

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::background::AmbientModel;
use crate::dataset::{GuideDataset, GuideObservation};
use crate::stats::{expected_ambient_given_true, posterior_real};

#[derive(Debug, Clone)]
pub struct FitConfig {
    pub max_iterations: usize,

    // Existing mathematical convergence.
    pub tolerance: f64,

    // Biological convergence.
    pub posterior_tolerance: f64,
    pub stable_iterations_required: usize,
    pub call_posterior_threshold: f64,

    pub initial_prior_real: f64,
    pub initial_dispersion: f64,
    pub minimum_true_mean: f64,
    pub minimum_lambda: f64,
    pub prior_alpha: f64,
    pub prior_beta: f64,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,

            tolerance: 1e-5,

            // Posterior probabilities may move by at most 0.001.
            posterior_tolerance: 1e-3,

            // Require biological stability several times in a row.
            stable_iterations_required: 5,

            // cutoff for calling 
            call_posterior_threshold: 0.95,

            initial_prior_real: 0.05,
            initial_dispersion: 10.0,
            minimum_true_mean: 0.5,
            minimum_lambda: 1e-6,
            prior_alpha: 0.5,
            prior_beta: 9.5,
        }
    }
}


#[derive(Debug, Clone)]
pub struct FitIteration {
    pub iteration: usize,

    pub max_parameter_delta: f64,

    pub max_posterior_delta: f64,
    pub mean_posterior_delta: f64,

    pub changed_calls: usize,
    pub total_observations: usize,
}

#[derive(Debug, Clone)]
pub struct GuideExpressionModel {
    /// P(a cell genuinely contains this guide).
    pub prior_real: f64,
    /// Mean true guide-expression count, excluding ambient molecules.
    pub mean: f64,
    /// NB size/dispersion parameter. Large values approach Poisson.
    pub theta: f64,
}

#[derive(Debug, Clone)]
pub struct ObservationFit {
    pub observation: GuideObservation,
    pub posterior_real: f64,
    pub expected_ambient: f64,
    pub expected_true: f64,
}

#[derive(Debug, Clone)]
pub struct FittedModel {
    pub ambient: AmbientModel,
    pub guides: Vec<GuideExpressionModel>,
    pub lambda_by_cell: HashMap<u64, f64>,
    pub observations: Vec<ObservationFit>,

    pub iterations: usize,

    pub mathematical_converged: bool,
    pub biological_converged: bool,

    pub diagnostics: Vec<FitIteration>,
}


pub fn fit_mixture(
    data: &GuideDataset,
    ambient: AmbientModel,
    cfg: &FitConfig,
) -> Result<FittedModel> {
    if data.n_guides() != ambient.guide_probability.len() {
        bail!("Guide count differs between ambient model and filtered data");
    }

    if data.n_cells() == 0 {
        bail!("Cannot fit guide model without filtered cells");
    }

    /*
     * Initial estimate:
     *
     * lambda_c = total observed guide burden in the cell.
     *
     * This deliberately overestimates ambient burden initially because
     * genuine guide molecules are still included. The EM iterations will
     * subsequently separate expected ambient and expected true-guide
     * molecules.
     */
    let mut lambda_by_cell: HashMap<u64, f64> = data
        .cell_ids
        .iter()
        .copied()
        .map(|cell_id| {
            (
                cell_id,
                (data.cell_total(cell_id) as f64)
                    .max(cfg.minimum_lambda),
            )
        })
        .collect();

    /*
     * Initial guide-specific true-expression models.
     */
    let mut guides =
        initialize_guides(
            data,
            &ambient,
            &lambda_by_cell,
            cfg,
        );

    /*
     * Flatten the guide-major observations.
     *
     * Their order never changes during fitting. This is important:
     * previous_posteriors[i] always refers to observations[i].
     */
    let mut observations: Vec<ObservationFit> = data
        .by_guide
        .iter()
        .flat_map(|guide| guide.iter().copied())
        .map(|observation| ObservationFit {
            observation,
            posterior_real: 0.0,
            expected_ambient: observation.count as f64,
            expected_true: 0.0,
        })
        .collect();

    /*
     * Biological convergence state.
     *
     * Empty means that no previous biological state exists yet.
     * After the first E-step this vector receives the first set of
     * posteriors.
     */
    let mut previous_posteriors: Vec<f64> = Vec::new();

    let mut stable_iterations = 0usize;

    let mut mathematical_converged = false;
    let mut biological_converged = false;

    let mut iterations = 0usize;

    let mut diagnostics = Vec::new();

    for iter in 0..cfg.max_iterations {
        iterations = iter + 1;

        /*
         * ============================================================
         * E STEP
         * ============================================================
         *
         * For every observed cell/guide pair calculate:
         *
         *   P(real guide | observed count)
         *
         * using
         *
         * ambient:
         *   A_cg ~ Poisson(lambda_c * p_g)
         *
         * true guide:
         *   T_cg ~ NB(mu_g, theta_g)
         *
         * genuine observation:
         *   X_cg = A_cg + T_cg
         */
        observations
            .par_iter_mut()
            .for_each(|state| {
                let obs = state.observation;

                let guide = &guides[obs.guide_id as usize];

                let lambda = lambda_by_cell[&obs.cell_id];

                let ambient_mean =
                    lambda * ambient.p(obs.guide_id);

                let posterior = posterior_real(
                    obs.count,
                    ambient_mean,
                    guide.prior_real,
                    guide.mean,
                    guide.theta,
                );

                /*
                 * Even when the guide is genuinely present, part of the
                 * observed count may still be ambient.
                 */
                let ambient_if_real =
                    expected_ambient_given_true(
                        obs.count,
                        ambient_mean,
                        guide.mean,
                        guide.theta,
                    );

                state.posterior_real = posterior;

                state.expected_ambient =
                    (1.0 - posterior)
                        * obs.count as f64
                    + posterior
                        * ambient_if_real;

                state.expected_true =
                    posterior
                        * (
                            obs.count as f64
                                - ambient_if_real
                        )
                        .max(0.0);
            });

        /*
         * ============================================================
         * BIOLOGICAL STABILITY
         * ============================================================
         *
         * Compare the current posterior assignment state against the
         * previous iteration BEFORE changing previous_posteriors.
         */
        let mut changed_calls = 0usize;
        let mut max_posterior_delta = f64::INFINITY;
        let mut mean_posterior_delta = f64::INFINITY;

        if !previous_posteriors.is_empty() {
            assert_eq!(
                previous_posteriors.len(),
                observations.len(),
                "number/order of observations changed during fitting"
            );

            let (changed, max_delta, sum_delta) = previous_posteriors
                .par_iter()
                .zip(observations.par_iter())
                .map(|(previous, state)| {
                    let current = state.posterior_real;
                    let delta = (current - *previous).abs();

                    let previous_call = *previous >= cfg.call_posterior_threshold;
                    let current_call = current >= cfg.call_posterior_threshold;

                    let changed = usize::from(previous_call != current_call);

                    (changed, delta, delta)
                })
                .reduce(
                    || (0usize, 0.0_f64, 0.0_f64),
                    |a, b| (
                        a.0 + b.0,
                        a.1.max(b.1),
                        a.2 + b.2,
                    ),
                );

            changed_calls = changed;
            max_posterior_delta = max_delta;
            mean_posterior_delta =
                sum_delta / observations.len() as f64;

            if changed_calls == 0 {
                stable_iterations += 1;
            } else {
                stable_iterations = 0;
            }

            if stable_iterations >= cfg.stable_iterations_required {
                biological_converged = true;
            }
        }

        /*eprintln!(
            "iter {:>3}: changed_calls={:>5}, stable={:>3}, max_post_delta={:.6e}",
            iter + 1,
            changed_calls,
            stable_iterations,
            max_posterior_delta,
        );*/

        /*
         * ONLY NOW replace the old biological state.
         *
         * Nothing below this point needs the previous iteration's
         * posterior vector.
         */
        previous_posteriors.clear();

        previous_posteriors.extend(
            observations
                .iter()
                .map(|state| state.posterior_real),
        );

        /*
         * ============================================================
         * SAVE OLD PARAMETERS
         * ============================================================
         *
         * Needed for mathematical convergence diagnostics after the
         * M-step.
         */
        let old_lambda =
            lambda_by_cell.clone();

        let old_guides =
            guides.clone();

        /*
         * ============================================================
         * M STEP: CELL-SPECIFIC AMBIENT BURDEN
         * ============================================================
         *
         * lambda_c becomes the expected number of ambient guide
         * molecules in that cell.
         */
        for value in lambda_by_cell.values_mut() {
            *value = cfg.minimum_lambda;
        }

        for state in &observations {
            *lambda_by_cell
                .entry(state.observation.cell_id)
                .or_insert(cfg.minimum_lambda)
                += state.expected_ambient;
        }

        /*
         * ============================================================
         * M STEP: GUIDE-SPECIFIC TRUE EXPRESSION
         * ============================================================
         */
        let n_cells =
            data.n_cells() as f64;

        for guide_id in 0..guides.len() {
            /*
             * Gather observations belonging to this guide.
             *
             * We keep the guide-major structure in GuideDataset, but the
             * EM observation state is flattened. With only a modest
             * number of guides this is fine for now.
             */
            let states: Vec<&ObservationFit> = observations
                .iter()
                .filter(|state| {
                    state.observation.guide_id as usize
                        == guide_id
                })
                .collect();

            /*
             * Expected number of genuinely positive cells for guide g.
             */
            let sum_z: f64 = states
                .iter()
                .map(|state| state.posterior_real)
                .sum();

            /*
             * Expected number of true guide molecules.
             */
            let sum_true: f64 = states
                .iter()
                .map(|state| state.expected_true)
                .sum();

            /*
             * Guide-specific probability that an arbitrary cell
             * genuinely contains this guide.
             *
             * Beta prior prevents pathological zero/one estimates.
             */
            let prior_real =
                (sum_z + cfg.prior_alpha)
                    / (
                        n_cells
                            + cfg.prior_alpha
                            + cfg.prior_beta
                    );

            /*
             * Mean true-guide expression conditional on genuine guide
             * presence.
             */
            let true_mean =
                if sum_z > 1e-8 {
                    (
                        sum_true / sum_z
                    )
                    .max(cfg.minimum_true_mean)
                } else {
                    cfg.minimum_true_mean
                };

            /*
             * Approximate NB dispersion using a posterior-weighted
             * method-of-moments estimate.
             *
             * If variance <= mean, the NB approaches Poisson, represented
             * here by a very large theta.
             */
            let mut weighted_variance = 0.0;

            if sum_z > 1e-8 {
                for state in &states {
                    let inferred_true =
                        if state.posterior_real > 1e-12 {
                            state.expected_true
                                / state.posterior_real
                        } else {
                            0.0
                        };

                    weighted_variance +=
                        state.posterior_real
                            * (
                                inferred_true
                                    - true_mean
                            )
                            .powi(2);
                }

                weighted_variance /= sum_z;
            }

            let theta =
                if weighted_variance
                    > true_mean + 1e-8
                {
                    (
                        true_mean * true_mean
                            / (
                                weighted_variance
                                    - true_mean
                            )
                    )
                    .clamp(0.05, 1e6)
                } else {
                    /*
                     * Very high theta means that the NB approaches a
                     * Poisson distribution.
                     */
                    1e6
                };

            guides[guide_id] =
                GuideExpressionModel {
                    prior_real:
                        prior_real.clamp(
                            1e-9,
                            1.0 - 1e-9,
                        ),
                    mean: true_mean,
                    theta,
                };
        }

        /*
         * ============================================================
         * MATHEMATICAL CONVERGENCE
         * ============================================================
         */
        let mut max_parameter_delta =
            0.0_f64;

        for (&cell_id, &new_value)
            in &lambda_by_cell
        {
            let old_value = old_lambda
                .get(&cell_id)
                .copied()
                .unwrap_or(new_value);

            max_parameter_delta =
                max_parameter_delta.max(
                    relative_delta(
                        old_value,
                        new_value,
                    ),
                );
        }

        for (old, new) in
            old_guides.iter().zip(&guides)
        {
            max_parameter_delta =
                max_parameter_delta.max(
                    relative_delta(
                        old.prior_real,
                        new.prior_real,
                    ),
                );

            max_parameter_delta =
                max_parameter_delta.max(
                    relative_delta(
                        old.mean,
                        new.mean,
                    ),
                );

            max_parameter_delta =
                max_parameter_delta.max(
                    relative_delta(
                        old.theta,
                        new.theta,
                    ),
                );
        }

        if max_parameter_delta < cfg.tolerance {
            mathematical_converged = true;
        }

        /*
         * Record both kinds of convergence.
         *
         * The first iteration has no previous biological state, hence
         * posterior deltas are infinity there.
         */
        diagnostics.push(
            FitIteration {
                iteration: iter + 1,
                max_parameter_delta,
                max_posterior_delta,
                mean_posterior_delta,
                changed_calls,
                total_observations:
                    observations.len(),
            },
        );

        /*
         * ============================================================
         * STOPPING CONDITION
         * ============================================================
         *
         * Either:
         *
         * 1. the underlying numerical parameters have converged, OR
         *
         * 2. the biological assignments/posteriors have remained stable
         *    for several consecutive iterations.
         *
         * This is deliberate: there is little value in spending another
         * hundred iterations polishing NB dispersion if the inferred
         * cell-guide biology is already invariant.
         */
        if mathematical_converged
            || biological_converged
        {
            break;
        }
    }

    /*
     * ================================================================
     * FINAL E STEP
     * ================================================================
     *
     * The final iteration above ends with an M-step.
     *
     * Therefore calculate posteriors once more so that every returned
     * ObservationFit corresponds exactly to the returned final model
     * parameters.
     */
    for state in &mut observations {
        let obs =
            state.observation;

        let guide =
            &guides[obs.guide_id as usize];

        let lambda =
            lambda_by_cell[&obs.cell_id];

        let ambient_mean =
            lambda * ambient.p(obs.guide_id);

        let posterior =
            posterior_real(
                obs.count,
                ambient_mean,
                guide.prior_real,
                guide.mean,
                guide.theta,
            );

        let ambient_if_real =
            expected_ambient_given_true(
                obs.count,
                ambient_mean,
                guide.mean,
                guide.theta,
            );

        state.posterior_real =
            posterior;

        state.expected_ambient =
            (1.0 - posterior)
                * obs.count as f64
            + posterior
                * ambient_if_real;

        state.expected_true =
            posterior
                * (
                    obs.count as f64
                        - ambient_if_real
                )
                .max(0.0);
    }

    Ok(FittedModel {
        ambient,
        guides,
        lambda_by_cell,
        observations,
        iterations,
        mathematical_converged,
        biological_converged,
        diagnostics,
    })
}

fn initialize_guides(
    data: &GuideDataset,
    ambient: &AmbientModel,
    lambda_by_cell: &HashMap<u64, f64>,
    cfg: &FitConfig,
) -> Vec<GuideExpressionModel> {
    data.by_guide
        .iter()
        .enumerate()
        .map(|(guide_id, obs)| {
            let mut excess: Vec<f64> = obs
                .iter()
                .map(|x| {
                    let ambient_mean =
                        lambda_by_cell[&x.cell_id] * ambient.p(guide_id as u32);
                    (x.count as f64 - ambient_mean).max(0.0)
                })
                .filter(|x| *x > 0.0)
                .collect();

            excess.sort_by(f64::total_cmp);
            let mean = if excess.is_empty() {
                cfg.minimum_true_mean
            } else {
                // Upper-half mean is deliberately resistant to the sea of
                // low-count ambient observations at initialization.
                let start = excess.len() / 2;
                (excess[start..].iter().sum::<f64>()
                    / (excess.len() - start) as f64)
                    .max(cfg.minimum_true_mean)
            };

            GuideExpressionModel {
                prior_real: cfg.initial_prior_real,
                mean,
                theta: cfg.initial_dispersion,
            }
        })
        .collect()
}

fn relative_delta(old: f64, new: f64) -> f64 {
    (old - new).abs() / old.abs().max(new.abs()).max(1e-9)
}

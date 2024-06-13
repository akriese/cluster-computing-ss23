use super::Body;
use super::G;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, Deserialize, Serialize)]
pub(crate) struct TreeNode {
    pub(crate) center: [f64; 2],
    pub(crate) size: f64,
    pub(crate) mass: f64,
    pub(crate) mass_center: [f64; 2],
    pub(crate) children: Vec<TreeNode>,
    pub(crate) body: Option<Body>,
}

impl TreeNode {
    /// Creates four subtrees as children for self.
    /// Each child represents one quadrant of the original tree span.
    pub(crate) fn split(&mut self) {
        let center_offset = self.size / 4_f64;
        let mut dummy = TreeNode {
            size: self.size / 2_f64,
            ..Default::default()
        };
        dummy.center = [
            self.center[0] + center_offset,
            self.center[1] + center_offset,
        ];
        self.children.push(dummy.clone());
        dummy.center = [
            self.center[0] - center_offset,
            self.center[1] + center_offset,
        ];
        self.children.push(dummy.clone());
        dummy.center = [
            self.center[0] - center_offset,
            self.center[1] - center_offset,
        ];
        self.children.push(dummy.clone());
        dummy.center = [
            self.center[0] + center_offset,
            self.center[1] - center_offset,
        ];
        self.children.push(dummy);
    }

    /// Push a given body down the hierarchy. If no chlidren exist yet, creates them.
    ///
    /// * `body`: Reference to a Body object.
    pub(crate) fn push_to_child(&mut self, body: &Body) {
        if self.children.is_empty() {
            self.split();
        }

        if body.position[0] > self.center[0] {
            if body.position[1] > self.center[1] {
                self.children[0].insert(body);
            } else {
                self.children[3].insert(body);
            }
        } else if body.position[1] > self.center[1] {
            self.children[1].insert(body);
        } else {
            self.children[2].insert(body);
        }
    }

    /// Insert a Body into the tree. The following three cases must be handled:
    ///
    /// 1. self is an empty leaf -> simply assign the body to self.body
    /// 2. self is a body leaf -> push down the existing and the given body
    /// 3. self has children already -> push down the given body
    ///
    /// * `body`: reference to the body to be inserted
    pub(crate) fn insert(&mut self, body: &Body) {
        if self.children.is_empty() && self.body.is_none() {
            self.body = Some(body.clone());
        } else {
            if let Some(b) = &self.body {
                self.push_to_child(&b.clone());
                self.body = None;
            }

            self.push_to_child(body);
        }

        // update the node's mass and mass_center
        self.mass += body.mass;
        self.mass_center[0] = (self.mass_center[0] * (self.mass - body.mass)
            + body.position[0] * body.mass)
            / self.mass;
        self.mass_center[1] = (self.mass_center[1] * (self.mass - body.mass)
            + body.position[1] * body.mass)
            / self.mass;
    }

    /// Recursively calculate the force between self and the given body.
    ///
    /// Theta is used as a threshold ratio for distance between self and the body. If
    /// they are far enough away form each other, self.mass and self.mass_center are
    /// are used for the force calculation which is the central point of Barnes-Hut.
    ///
    /// * `body`: The body to calculate the force to.
    /// * `theta`: Threshold ratio parameter for shortcutting the calculation.
    pub(crate) fn calculate_force(&self, body: &Body, theta: f64) -> [f64; 2] {
        let displacement = [
            self.mass_center[0] - body.position[0],
            self.mass_center[1] - body.position[1],
        ];
        let distance =
            (displacement[0] * displacement[0] + displacement[1] * displacement[1]).sqrt();

        // avoid massive forces when bodies are super close to each other
        if distance < 1e-10f64 {
            return [0f64; 2];
        }

        if let Some(b) = &self.body {
            let f = G * b.mass * body.mass / (distance * distance);
            [
                f + displacement[0] / distance,
                f + displacement[1] / distance,
            ]
        } else if !self.children.is_empty() {
            if self.size / distance < theta {
                let f = G * self.mass * body.mass / (distance * distance);
                [
                    f + displacement[0] / distance,
                    f + displacement[1] / distance,
                ]
            } else {
                let mut summed_force = [f64::default(); 2];
                for child in self.children.iter() {
                    let f = child.calculate_force(body, theta);
                    summed_force[0] += f[0];
                    summed_force[1] += f[1];
                }

                summed_force
            }
        } else {
            // empty quadrant
            [0f64; 2]
        }
    }

    /// Merge to trees, consuming the given tree.
    ///
    /// * `other`: Another tree to be merged into self.
    pub(crate) fn merge(&mut self, mut other: TreeNode, depth: usize) {
        // this assumes that two trees with the same size and center get merged
        assert!(self.size == other.size);
        assert!(self.center == other.center);

        if let Some(body) = &self.body {
            // 1. case: self is single body
            other.insert(body);
            *self = other;
        } else if self.children.is_empty() {
            // 2. case: self is empty
            *self = other;
        } else {
            // 3. case: self has children
            // needs to handle three cases for other
            if let Some(b) = &other.body {
                self.insert(b);
            } else if other.children.is_empty() {
                // empty case, other is empty quadrant and nothing to do here...
            } else {
                if depth > 3 {
                    for (self_child, other_child) in
                        self.children.iter_mut().zip(other.children.into_iter())
                    {
                        self_child.merge(other_child, depth + 1);
                    }
                } else {
                    let ss = self.children.par_iter_mut();
                    let os = other.children.into_par_iter();
                    ss.zip(os).for_each(|(s, o)| s.merge(o, depth + 1));
                }

                self.mass = self.children.iter().map(|c| c.mass).sum();
                self.mass_center[0] = self
                    .children
                    .iter()
                    .map(|c| c.mass_center[0] * c.mass)
                    .sum::<f64>()
                    / self.mass;
                self.mass_center[1] = self
                    .children
                    .iter()
                    .map(|c| c.mass_center[1] * c.mass)
                    .sum::<f64>()
                    / self.mass;
            }
        }
    }

    pub(crate) fn height(&self) -> usize {
        return if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.height()).max().unwrap()
        };
    }
}

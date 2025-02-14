
/*
 * Author: William Wadsworth
 * Date: 6.14.2024
 *
 * About:
 *    This is the implementation file for the Rust geometry class
*/

use std::f64::consts::PI;

pub struct Geometry;

impl Geometry {
    /// Calculates the distance between two Cartesian coordinates (x1, y1, x2, y2).
    pub fn distance(q: f64, w: f64, e: f64, r: f64) -> f64 {
        ((e-q).powi(2) + (r-w).powi(2)).sqrt()
    }

    /// Calculates the radius between two Cartesian coordinates (x1, y1, x2, y2).
    pub fn radius(a: f64, s: f64, d: f64, f: f64) -> f64 {
        Self::distance(a,s,d,f)
    }

    /// Calculates the circumference of a circle.
    pub fn circumference(radius: f64) -> f64 {
        2.0*PI*radius
    }

    /// Calculates the area of a circle.
    pub fn area_circle(radius: f64) -> f64 {
        PI*radius.powi(2)
    }

    /// Determines if three sides of a triangle make a right triangle.
    pub fn is_right_triangle(a: f64, b: f64, c: f64) -> bool {
        (a*a + b*b - c*c).abs() < f64::EPSILON
    }
}

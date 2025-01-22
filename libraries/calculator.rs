//use std::f64::consts::PI;

/// A simple calculator struct with various mathematical methods.
pub struct Calculator;

impl Calculator {
    /// Returns the factorial or double factorial of a number.
    pub fn factorial(endpoint: usize, double_factorial: bool) -> Result<usize, String> {
        if endpoint > 1000 {
            return Err("Invalid endpoint, integer must be between 0 and 1,000".to_string());
        }

        let mut prod: usize = 1;

        if double_factorial {
            if endpoint % 2 == 0 {
                return Err("Invalid endpoint for double factorial, endpoint must be odd".to_string());
            }
            for i in (1..=endpoint).step_by(2) {
                prod *= i;
            }
        } else {
            for i in 1..=endpoint {
                prod *= i;
            }
        }

        Ok(prod)
    }

    /// Returns the geometric series of a number.
    pub fn geoseries(initial_term: f64, num_terms: usize, r: f64) -> Result<f64, String> {
        if r <= 0.0 {
            return Err("Common ratio must be positive.".to_string());
        }

        let mut series_sum = 0.0;
        let mut term = initial_term;

        for _ in 0..num_terms {
            series_sum += term;
            term *= r;
        }

        Ok(series_sum)
    }

    /// Returns the result of a basic arithmetic operation.
    pub fn four_function(
        operand_1: f64,
        operand_2: f64,
        operation: &str,
    ) -> Result<f64, String> {
        match operation.to_lowercase().as_str() {
            "add" => Ok(operand_1 + operand_2),
            "subtract" => Ok(operand_1 - operand_2),
            "multiply" => Ok(operand_1 * operand_2),
            "divide" => {
                if operand_2 == 0.0 {
                    Err("Error: cannot divide by 0".to_string())
                } else {
                    Ok(operand_1 / operand_2)
                }
            }
            _ => Err("Error: invalid operation".to_string()),
        }
    }

    /// Determines if one number is a multiple of another and returns its multiples.
    pub fn is_multiple(
        max_val: f64,
        increment: f64,
    ) -> Result<(bool, usize, Vec<f64>), String> {
        if increment == 0.0 {
            return Err("Increment must not be zero".to_string());
        }

        let mut multiples = Vec::new();
        let mut current_val = increment;

        while current_val < max_val {
            multiples.push(current_val);
            current_val += increment;
        }

        Ok((!multiples.is_empty(), multiples.len(), multiples))
    }

    /// Solves the quadratic equation.
    pub fn quadratic(a: f64, b: f64, c: f64) -> Result<(f64, f64), String> {
        if a == 0.0 {
            return Err("The value of 'a' cannot be zero in a quadratic equation.".to_string());
        }

        let discriminant = b * b - 4.0 * a * c;

        if discriminant > 0.0 {
            let root1 = (-b + discriminant.sqrt()) / (2.0 * a);
            let root2 = (-b - discriminant.sqrt()) / (2.0 * a);
            Ok((root1, root2))
        } else if discriminant == 0.0 {
            let root = -b / (2.0 * a);
            Ok((root, root))
        } else {
            let real_part = -b / (2.0 * a);
            let imaginary_part = (discriminant.abs().sqrt()) / (2.0 * a);
            Err(format!(
                "Complex roots: {} ± {}i",
                real_part, imaginary_part
            ))
        }
    }
}


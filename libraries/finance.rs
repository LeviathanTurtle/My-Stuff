
/*
 * Author: William Wadsworth
 * Date: 2.13.2025
 *
 * About:
 *    This is the implementation file for the Rust finance class
*/

use std::fs::File;
use std::io::Write;

pub struct Finance;

impl Finance {
    /// Returns the minimum number of coins given a monetary value.
    pub fn find_coin_total(total: f64) -> Result<(i32, i32, i32, i32), String> {
        if total <= 0.0 {
            return Err("Invalid total, amount must be greater than 0".to_string());
        }

        let mut total = (total * 100.0).round() as i32;

        let quarters = total / 25;
        total %= 25;
        let dimes = total / 10;
        total %= 10;
        let nickels = total / 5;
        let pennies = total % 5;

        Ok((quarters, dimes, nickels, pennies))
    }

    /// Generates an investment table.
    pub fn gen_investment_table(
        principal_amount: f64,
        interest_rate: f64,
        time: f64,
        deposit: f64,
        interest_rate_change: f64,
        output: &str,
    ) -> Result<(), String> {
        if ![principal_amount, interest_rate, time, deposit, interest_rate_change]
            .iter()
            .all(|&x| x.is_finite())
        {
            return Err("All parameters must be numeric types (int or float)".to_string());
        }

        let time_months = (time * 12.0) as i32;
        let mut t = 1;
        let mut value_of_investment = 0.0;
        let mut count = 0;

        let mut investment_table = format!(
            "{:^35}\n{:>8} | {:>18} | {:>22}\n{}\n",
            "Investment Table", "Month", "Total Invested ($)", "Value of Investment ($)", "-".repeat(50)
        );

        while t <= time_months {
            value_of_investment = principal_amount + (principal_amount * interest_rate * (t as f64) / 12.0) + ((t as f64) * deposit);
            investment_table.push_str(&format!("{:8} | {:18.2} | {:22.2}\n", t, (t as f64) * deposit, value_of_investment));
            t += 1;

            if interest_rate_change != 0.0 {
                count += 1;
                if count % 12 == 0 {
                    interest_rate += interest_rate_change;
                }
            }
        }

        let end_msg = format!("\nYour capital gain will be ${:.2} in {:.2} years\n", value_of_investment - principal_amount, time);

        if output == "CONSOLE" {
            println!("{}{}", investment_table, end_msg);
        } else {
            let mut file = File::create("investment_table").map_err(|_| "Failed to create file")?;
            file.write_all(format!("{}{}", investment_table, end_msg).as_bytes())
                .map_err(|_| "Failed to write to file")?;
        }

        Ok(())
    }

    /// Calculate total money based on USD denominations.
    pub fn money_calculator(count_1: i32, count_5: i32, count_10: i32, count_20: i32, count_50: i32, count_100: i32) -> Result<i32, String> {
        let counts = [count_1, count_5, count_10, count_20, count_50, count_100];
        if !counts.iter().all(|&x| x >= 0) {
            return Err("All denomination counts must be non-negative integers".to_string());
        }
        Ok(count_1 + (5 * count_5) + (10 * count_10) + (20 * count_20) + (50 * count_50) + (100 * count_100))
    }
}

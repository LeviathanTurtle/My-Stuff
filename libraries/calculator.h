/*
 * Author: William Wadsworth
 * Date: 6.12.2024
 *
 * About:
 *    This is the header file for the calculator class
*/

#ifndef CALCULATOR
#define CALCULATOR

#include <string>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <type_traits>  // static_assert

class calculator {
    private:
        double a, b, c;

    public:
        void set_a(const double&);
        void set_b(const double&);
        void set_c(const double&);
        double get_a();
        double get_b();
        double get_c();

        long int factorial(const int&, const bool&);
        double geoseries(double, const int&, double);
        
        /* function that returns the sum, difference, product, or quotient of two numbers (supports
         * only numerical types). 
         * pre-condition: operand_1 and operand_2 parameters must be initialized with values. If
         *                dividing, operand_2 cannot be 0. operation parameter must be initialized 
         *                to a non-empty string
         * 
         * post-condition: depending on the operation specified (assuming the operation is valid), 
         *                 the sum, difference, product, or quotient is returned, otherwise an 
         *                 error is output and a relevant exception is thrown
        */
        template <typename T>
        T fourFunction(const T& operand_1, const T& operand_2, std::string operation)
        {
            // ensure template supports arithmetic operations
            static_assert(std::is_arithmetic<T>::value, "Operands must be an arithmetic type");

            // convert string to lowercase
            std::transform(operation.begin(), operation.end(), operation.begin(), ::tolower);

            if (operation == "add")
                return operand_1 + operand_2;
            else if (operation == "subtract")
                return operand_1 - operand_2;
            else if (operation == "multiply")
                return operand_1 * operand_2;
            else if (operation == "divide")
                if (operand_2 == 0) {
                    //std::cerr << "Error: cannot divide by 0\n";
                    throw std::invalid_argument("Error: cannot divide by 0");
                }
                return operand_1 / operand_2;
            else {
                //std::cerr << "Error: invalid operation\n";
                throw std::invalid_argument("Error: invalid operation");
            }
        }

        /* function to determine if one number is a multiple of another (and its multiples)
         * pre-condition: max and increment must be initialized to positive non-zero numerical
         *                values, count must be initialized (will be reset to 0 in case of invalid
         *                value), multiples must be intialized to an empty vector
         * 
         * post-condition: if there are no multiples, false is returned, otherwise true is returned
         *                 and the count parameter is updated with however many multiples there are
         *                 as well as the 'multiples' vector paramter being updated with each
         *                 multiple
        */
        template <typename T>
        bool isMultiple(const T& max, const T& increment, int& count, std::vector<T>& multiples)
        {
            // sum var for calculation
            T sum=0;

            // ensure count starts at 0
            if (count != 0)
                count = 0;

            // check if x is divisible by y
            if( max % increment == 0) {
                // calculation of multiples
                while(sum < max) {
                    sum += increment;
                    multiples.push_back(sum);
                    count++;
                }

                return true;
            } else return false;
        }


};

#endif


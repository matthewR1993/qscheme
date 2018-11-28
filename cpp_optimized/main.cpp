using namespace std;
#include <iostream>
#include <complex>
#include <chrono>
#include <random>
#include <string>


const size_t sz = 10;

//Constants and utils.
const complex<double> I(0.0,1.0);
complex<double> fact_arr[sz*4];
complex<double> fact_sqrt_arr[sz*4];

double factorial(int n){
    double m = n;

    if (n <= 1){
        return 1;
    } else {
        for (int i = n - 1; i > 0; --i) {
            m = m * i;
        }
        return m;
    }
}

void show_time(
        std::chrono::high_resolution_clock::time_point start,
        std::chrono::high_resolution_clock::time_point finish
) {
    chrono::duration<double> elapsed = finish - start;
    cout << "Elapsed time: " << elapsed.count() << " s\n";
}


// Core functions.

//complex<double> rho_in [sz][sz][sz][sz];
template <size_t dim>
void build_coherent_state(complex<double> alpha, complex<double> (&arr)[dim]){
    for (size_t k = 0; k < dim; ++k) {
        arr[k] = k;
    }
}

template <size_t dim>
void build_fock_state(size_t n, complex<double> (&arr)[dim]){
    arr[n] = 1;
}

template <size_t dim>
void build_bs_parameters(double T_min, double T_max, complex<double> (&arr)[dim]){
    arr[1] = 0;
}

// Density matrix transformation at BS.
// Takes: applied density matrix in 2 channels.
// Returns: applied density matrix in 2 channels.
template <size_t dim>
void bs_matrix_transform_opt(complex<double> (&input_matrix)[dim][dim][dim][dim], complex<double> (&output_matrix)[dim*2][dim*2][dim*2][dim*2], double t, double r)
{
    size_t d1;
    size_t d2;
    size_t d1_;
    size_t d2_;
    complex<double> coeff1;
    complex<double> coeff2;
    complex<double> tc = t;
    complex<double> rc = r;

    for (size_t p1 = 0; p1 < dim; ++p1) {
        for (size_t p1_ = 0; p1_ < dim; ++p1_) {
            for (size_t p2 = 0; p2 < dim; ++p2) {
                for (size_t p2_ = 0; p2_ < dim; ++p2_) {

                    for (size_t n = 0; n < p1 + 1; ++n) {
                        for (size_t k = 0; k < p2 + 1; ++k) {
                            for (size_t n_ = 0; n_ < p1_ + 1; ++n_) {
                                for (size_t k_ = 0; k_ < p2_ + 1; ++k_) {
                                    d1 = p1 - n + k;
                                    d2 = n + p2 - k;
                                    coeff1 = pow(tc, p1 - n + p2 - k) * pow(I * rc, n + k) * fact_sqrt_arr[d1] * fact_sqrt_arr[d2] * fact_sqrt_arr[p1] * fact_sqrt_arr[p2] / (fact_arr[n] * fact_arr[p1 - n] * fact_arr[k] * fact_arr[p2 - k]);

                                    d1_ = p1_ - n_ + k_;
                                    d2_ = n_ + p2_ - k_;
                                    coeff2 = pow(tc, p1_ - n_ + p2_ - k_) * pow(-I * rc, n_ + k_) * fact_sqrt_arr[d1_] * fact_sqrt_arr[d2_] * fact_sqrt_arr[p1_] * fact_sqrt_arr[p2_] / (fact_arr[n_] * fact_arr[p1_ - n_] * fact_arr[k_] * fact_arr[p2_ - k_]);

                                    output_matrix[d1][d2][d1_][d2_] = output_matrix[d1][d2][d1_][d2_] + input_matrix[p1][p2][p1_][p2_] * coeff1 * coeff2;
                                }
                            }
                        }
                    }

                }
            }
        }
    }
}


int main() {
    auto start = std::chrono::high_resolution_clock::now();

    char det = 'F';

    for (int i = 0; i < sz*4; ++i) {
        fact_arr[i] = factorial(i);
        fact_sqrt_arr[i] = sqrt(factorial(i));
    }

    //dm = evaluate_system();

    auto finish = chrono::high_resolution_clock::now();
    show_time(start, finish);
//    chrono::duration<double> elapsed = finish - start;
//    cout << "Elapsed time: " << elapsed.count() << " s\n";

    cout << "shit" << endl;
    return 0;
}


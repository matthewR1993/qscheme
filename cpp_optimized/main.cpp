using namespace std;
#include <iostream>
#include <complex>
#include <chrono>
#include <random>
#include <string>
#include <cmath>
#include <map>
#include <stdexcept>
#include <cstring>
#include <fstream>

#define _USE_MATH_DEFINES

//
// Setup constants.
//
// Series length.
const int L = 14;
// Density matrix size.
const int dim_dm = 10;

//
// Utils.
//
const size_t sz = 40;
const complex<double> I(0.0,1.0);
static complex<double> fact_arr[sz];
static complex<double> fact_sqrt_arr[sz];


void show_time(
        std::chrono::high_resolution_clock::time_point start,
        std::chrono::high_resolution_clock::time_point finish
) {
    chrono::duration<double> elapsed = finish - start;
    cout << "Elapsed time: " << elapsed.count() << " s\n";
}

//
// Core functions.
//
template <size_t dim>
void build_coherent_state(complex<double> alpha, complex<double> (&arr)[dim]){
    for (int k = 0; k < dim; ++k) {
        arr[k] = exp(- 0.5 * pow(abs(alpha), 2)) * pow(alpha, k) / fact_arr[k];
    }
}


template <size_t dim>
void build_fock_state(size_t n, complex<double> (&arr)[dim]){
    arr[n] = 1 / sqrt(tgamma(n + 1));
}


template <size_t dim>
void build_bs_parameters(double T_min, double T_max, double (&t_arr)[dim], double (&r_arr)[dim]){
    if (dim == 1) {
        t_arr[0] = sqrt(T_min);
        r_arr[0] = sqrt(1 - T_min);
    } else {
        double delta = (T_max - T_min) / (dim - 1);
        t_arr[dim - 1] = sqrt(T_max);
        r_arr[dim - 1] = sqrt(1 - T_max);
        for (size_t i = 0; i < dim - 1; ++i) {
            t_arr[i] = sqrt(T_min + delta * i);
            r_arr[i] = sqrt(1 - T_min - delta * i);
        }
    }
}


//Two channels (2x2) beam splitter transformation.
//With: t^2 + r^2 = 1.
//t: Transmission coefficient.
//r: Reflection coefficient.
//input_state: Unapplied state in two channels(modes).
//return: Transformed unapplied state in two channels(modes).
template <size_t dim>
void bs2x2_transform(
        double t,
        double r,
        complex<double> input_state[dim][dim],
        complex<double> (&output_state)[dim*2][dim*2]
){
    int ind1;
    int ind2;
    for (int n1 = 0; n1 < dim; ++n1) {
        for (int n2 = 0; n2 < dim; ++n2) {
            for (int k = 0; k < n1 + 1; ++k) {
                for (int l = 0; l < n2 + 1; ++l) {
                    ind1 = n1 - k + l;  // first channel index
                    ind2 = k + n2 - l;  // second channel index
                    output_state[ind1][ind2] = output_state[ind1][ind2] + input_state[n1][n2] * pow(I*r, k + l) * pow(t, n1 - k + n2 - l) * fact_arr[n1] * fact_arr[n2] / (fact_arr[k] * fact_arr[n1 - k] * fact_arr[l] * fact_arr[n2 - l]);
                }
            }
        }
    }
}


//Transformation at 2 beam splitters.
//Two input channels and four output channles - 2x4 transformation.
//Creation operators transformation:
//a1 => t1 a2 + i r1 a1.
//a2 => t2 a4 + i r2 a3.
//With transmission and reflection coefficients:
//t1^2 + r1^2 = 1.
//t2^2 + r2^2 = 1.
//:param t1: BS1 transmission.
//:param r1: BS1 reflection.
//:param t2: BS2 transmission.
//:param r2: BS2 reflection.
//:param input_state: Two channels(modes) unapllied state.
//:param output_state: Two channels(modes) unapllied state.
template <size_t dim>
void two_bs2x4_transform(
        double t1,
        double r1,
        double t2,
        double r2,
        complex<double> (&input_state)[dim][dim],
        complex<double> (&output_state)[dim][dim][dim][dim]
){
    int ind1;
    int ind2;
    int ind3;
    int ind4;
    for (int n1 = 0; n1 < dim; ++n1) {
        for (int n2 = 0; n2 < dim; ++n2) {
            for (int k = 0; k < n1 + 1; ++k) {
                for (int l = 0; l < n2 + 1; ++l) {
                    ind1 = k;
                    ind2 = n1 - k;
                    ind3 = l;
                    ind4 = n2 - l;
                    output_state[ind1][ind2][ind3][ind4] = output_state[ind1][ind2][ind3][ind4] + input_state[n1][n2] * pow(t1, n1 - k) * pow(I*r1, k) * pow(t2, n2 - l) * pow(I*r2, l) * fact_arr[n1] * fact_arr[n2] / (fact_arr[k] * fact_arr[n1 - k] * fact_arr[l] * fact_arr[n2 - l]);
                }
            }
        }
    }
}


//Tranformation of the state with POVM operator.
//:param input_state: Applied/unapplied state in 4 channels(modes).
//:param output_state: Applied/unapplied state in 4 channels(modes).
//:param detection_event: Detection event.
template <size_t dim>
void detection(
        char det_event,
        complex<double> (&input_state)[dim][dim][dim][dim],
        complex<double> (&output_state)[dim][dim][dim][dim]
){
    switch(det_event) {
        case 'F' :
            for (size_t p1 = 0; p1 < dim; ++p1) {
                for (size_t p2 = 0; p2 < dim; ++p2) {
                    for (size_t p3 = 0; p3 < dim; ++p3) {
                        for (size_t p4 = 0; p4 < dim; ++p4) {
                            if (p1 > 0 and p3 == 0)
                            {
                                output_state[p1][p2][p3][p4] = input_state[p1][p2][p3][p4];
                            }
                        }
                    }
                }
            }
            break;
        case 'T' :
            for (size_t p1 = 0; p1 < dim; ++p1) {
                for (size_t p2 = 0; p2 < dim; ++p2) {
                    for (size_t p3 = 0; p3 < dim; ++p3) {
                        for (size_t p4 = 0; p4 < dim; ++p4) {
                            if (p1 == 0 and p3 > 0)
                            {
                                output_state[p1][p2][p3][p4] = input_state[p1][p2][p3][p4];
                            }
                        }
                    }
                }
            }
            break;
        case 'B' :
            for (size_t p1 = 0; p1 < dim; ++p1) {
                for (size_t p2 = 0; p2 < dim; ++p2) {
                    for (size_t p3 = 0; p3 < dim; ++p3) {
                        for (size_t p4 = 0; p4 < dim; ++p4) {
                            if (p1 != 0 and p3 != 0)
                            {
                                output_state[p1][p2][p3][p4] = input_state[p1][p2][p3][p4];
                            }
                        }
                    }
                }
            }
            break;
        case 'N' :
            for (size_t p1 = 0; p1 < dim; ++p1) {
                for (size_t p2 = 0; p2 < dim; ++p2) {
                    for (size_t p3 = 0; p3 < dim; ++p3) {
                        for (size_t p4 = 0; p4 < dim; ++p4) {
                            if (p1 == 0 and p3 == 0)
                            {
                                output_state[p1][p2][p3][p4] = input_state[p1][p2][p3][p4];
                            }
                        }
                    }
                }
            }
            break;
        default:
            throw invalid_argument("Invalid parameter");
    }
}


//Calculating a probability of an event realisation.
//:param input_state: Unapplied state in 4 channels.
//:param detection_event: Detection event.
//:return: Probability of the detection.
template <size_t dim>
double det_probability(char det_event, complex<double> (&state)[dim][dim][dim][dim]){
    static complex<double> state_aft_det[dim][dim][dim][dim];
    detection(det_event, state, state_aft_det);

    complex<double> sum = 0.0;
    for (size_t p1 = 0; p1 < dim; ++p1) {
        for (size_t p2 = 0; p2 < dim; ++p2) {
            for (size_t p3 = 0; p3 < dim; ++p3) {
                for (size_t p4 = 0; p4 < dim; ++p4) {
                    sum = sum + state[p1][p2][p3][p4] * fact_arr[p1] * fact_arr[p2] * fact_arr[p3] * fact_arr[p4] * conj(state_aft_det[p1][p2][p3][p4]);
                }
            }
        }
    }
    return real(sum);
}


// Calculate the norm of the state.
//:param state: Unapplied state in 4 channels.
//:return: Norm of the state in 4 channels.
template <size_t dim>
double state_norm(complex<double> (&state)[dim][dim][dim][dim]){
    complex<double> sum = 0.0;
    for (size_t p1 = 0; p1 < dim; ++p1) {
        for (size_t p2 = 0; p2 < dim; ++p2) {
            for (size_t p3 = 0; p3 < dim; ++p3) {
                for (size_t p4 = 0; p4 < dim; ++p4) {
                    sum = sum + norm(state[p1][p2][p3][p4]) * fact_arr[p1] * fact_arr[p2] * fact_arr[p3] * fact_arr[p4];
                }
            }
        }
    }
    return sqrt(real(sum));
}


template <size_t dim>
void renormalize_state(
        double norm,
        complex<double> (&input_state)[dim][dim][dim][dim],
        complex<double> (&output_state)[dim][dim][dim][dim]
){
    for (size_t n1 = 0; n1 < dim; ++n1) {
        for (size_t n2 = 0; n2 < dim; ++n2) {
            for (size_t n3 = 0; n3 < dim; ++n3) {
                for (size_t n4 = 0; n4 < dim; ++n4) {
                    output_state[n1][n2][n3][n4] = input_state[n1][n2][n3][n4] / norm;
                }
            }
        }

    }
}


//Composing density matrix from projected vectors and partially trace it.
//:param input_state: Unapplied state in 4 channels.
//:return dm: Unapplied density matrix for 2 channels.
template <size_t state_dim, size_t dm_dim>
void build_dm_with_trace(
        complex<double> (&input_state)[state_dim][state_dim][state_dim][state_dim],
        complex<double> (&dm)[dm_dim][dm_dim][dm_dim][dm_dim]
){
    complex<double> dm_sum;
    for (size_t n2 = 0; n2 < dm_dim; ++n2) {
        for (size_t n4 = 0; n4 < dm_dim; ++n4) {
            for (size_t n2_ = 0; n2_ < dm_dim; ++n2_) {
                for (size_t n4_ = 0; n4_ < dm_dim; ++n4_) {
                    dm_sum = 0.0;
                    for (size_t p1 = 0; p1 < state_dim; ++p1) {
                        for (size_t p3 = 0; p3 < state_dim; ++p3) {
                            dm_sum += input_state[p1][n2][p3][n4] * conj(input_state[p1][n2_][p3][n4_]) * fact_arr[p1] * fact_arr[p3];
                        }
                    }
                    // unapplied case:
                    //dm[n2][n4][n2_][n4_] = dm_sum;

                    // applied case:
                    dm[n2][n4][n2_][n4_] = dm_sum * fact_sqrt_arr[n2] * fact_sqrt_arr[n4] * fact_sqrt_arr[n2_] * fact_sqrt_arr[n4_];
                }
            }
        }
    }
}


//A phase modulation for the density matrix in 2 channels.
//:param input_matrix: Input density matrix in 2 channels.
//:param output_matrix: Input density matrix in 2 channels.
//:param phase: Phase.
//:param channe: Channel (1 or 2).
template <size_t dim>
void phase_modulation(
        int channel,
        double phase_diff,
        complex<double> (&input_matrix)[dim][dim][dim][dim],
        complex<double> (&output_matrix)[dim][dim][dim][dim]
){
    double ind;
    switch(channel) {
        case 1 :
            for (int p1 = 0; p1 < dim; ++p1) {
                for (int p2 = 0; p2 < dim; ++p2) {
                    for (int p1_ = 0; p1_ < dim; ++p1_) {
                        for (int p2_ = 0; p2_ < dim; ++p2_) {
                            ind = p1 - p1_;
                            output_matrix[p1][p2][p1_][p2_] = input_matrix[p1][p2][p1_][p2_] * exp(I * phase_diff * ind);
                        }
                    }
                }
            }
            break;
        case 2 :
            for (int p1 = 0; p1 < dim; ++p1) {
                for (int p2 = 0; p2 < dim; ++p2) {
                    for (int p1_ = 0; p1_ < dim; ++p1_) {
                        for (int p2_ = 0; p2_ < dim; ++p2_) {
                            ind = p2 - p2_;
                            output_matrix[p1][p2][p1_][p2_] = input_matrix[p1][p2][p1_][p2_] * exp(I * phase_diff * ind);
                        }
                    }
                }
            }
            break;
        default:
            throw invalid_argument("Invalid parameter");
    }
}

//
// Squeezing.
//
//Average value of a coordinate quadrature:
//<X> = <(a + conj(a))/2>
//:param dm: Applied density matrix in 2 channels.
//:param channel: Number of the channel.
//:return: Average value of coordinate quadrature: <X> = <(a + conj(a))/2>
template <size_t dim>
complex<double> coord_aver(complex<double> (&dm)[dim][dim][dim][dim], int channel){
    complex<double> sum1, sum2;
    switch(channel) {
        case 1 :
            for (int m = 1; m < dim; ++m) {
                for (int n = 0; n < dim; ++n) {
                    sum1 += sqrt(m) * dm[m][n][m - 1][n];
                }
            }
            for (int m = 0; m < dim - 1; ++m) {
                for (int n = 0; n < dim; ++n) {
                    sum2 += sqrt(m + 1) * dm[m][n][m + 1][n];
                }
            }
            break;
        case 2 :
            for (int m = 0; m < dim; ++m) {
                for (int n = 1; n < dim; ++n) {
                    sum1 += sqrt(n) * dm[m][n][m][n - 1];
                }
            }
            for (int m = 0; m < dim; ++m) {
                for (int n = 0; n < dim - 1; ++n) {
                    sum2 += sqrt(n + 1) * dm[m][n][m][n + 1];
                }
            }
            break;
        default:
            throw invalid_argument("Invalid parameter");
    }
    return 0.5 * (sum1 + sum2);
}


//The average value of an impulse quadrature:
//<P> = <(a - conj(a))/2j>
//:param dm: Applied density matrix in 2 channels.
//:param channel: Number of the channel.
//:return: Average value of an impulse quadrature: <P> = <(a - conj(a))/2j>
template <size_t dim>
complex<double> impulse_aver(complex<double> (&dm)[dim][dim][dim][dim], int channel){
    complex<double> sum1, sum2;
    switch(channel) {
        case 1 :
            for (int m = 1; m < dim; ++m) {
                for (int n = 0; n < dim; ++n) {
                    sum1 += sqrt(m) * dm[m][n][m - 1][n];
                }
            }
            for (int m = 0; m < dim - 1; ++m) {
                for (int n = 0; n < dim; ++n) {
                    sum2 += sqrt(m + 1) * dm[m][n][m + 1][n];
                }
            }
            break;
        case 2 :
            for (int m = 0; m < dim; ++m) {
                for (int n = 1; n < dim; ++n) {
                    sum1 += sqrt(n) * dm[m][n][m][n - 1];
                }
            }
            for (int m = 0; m < dim; ++m) {
                for (int n = 0; n < dim - 1; ++n) {
                    sum2 += sqrt(n + 1) * dm[m][n][m][n + 1];
                }
            }
            break;
        default:
            throw invalid_argument("Invalid parameter");
    }
    return (0.5 / I) * (sum1 - sum2);
}


//Average value of coordinate quadrature product in 2 channels.
//:param dm: Applied density matrix in 2 channels.
//:return: Average value of coordinate quadrature product in 2 channels:
//<X1*X2> = (1/4) * <(a1 + conj(a1))*(a2 + conj(a2))>
template <size_t dim>
complex<double> prod_coord_aver(complex<double> (&dm)[dim][dim][dim][dim]){
    complex<double> sum1, sum2, sum3, sum4;
    // a1 * a2
    for (int m = 1; m < dim; ++m) {
        for (int n = 1; n < dim; ++n) {
            sum1 += sqrt(m * n) * dm[m][n][m - 1][n - 1];
        }
    }
    // a1 * conj(a2)
    for (int m = 1; m < dim; ++m) {
        for (int n = 0; n < dim - 1; ++n) {
            sum2 += sqrt(m * (n + 1)) * dm[m][n][m - 1][n + 1];
        }
    }
    // conj(a1) * a2
    for (int m = 0; m < dim - 1; ++m) {
        for (int n = 1; n < dim; ++n) {
            sum3 += sqrt((m + 1) * n) * dm[m][n][m + 1][n - 1];
        }
    }
    // conj(a1) * conj(a2)
    for (int m = 0; m < dim - 1; ++m) {
        for (int n = 0; n < dim - 1; ++n) {
            sum4 += sqrt((m + 1) * (n + 1)) * dm[m][n][m + 1][n + 1];
        }
    }
    return 0.25 * (sum1 + sum2 + sum3 + sum4);
}


//Average value of coordinate quadrature product in 2 channels.
//:param dm: Applied density matrix in 2 channels.
//:return:  Average value of coordinate quadrature product in 2 channels:
//<P1*P2> = (-1/4) * <(a1 - conj(a1))*(a2 - conj(a2))>
template <size_t dim>
complex<double> prod_impulse_aver(complex<double> (&dm)[dim][dim][dim][dim]){
    complex<double> sum1, sum2, sum3, sum4;
    // a1 * a2
    for (int m = 1; m < dim; ++m) {
        for (int n = 1; n < dim; ++n) {
            sum1 += sqrt(m * n) * dm[m][n][m - 1][n - 1];
        }
    }
    // a1 * conj(a2)
    for (int m = 1; m < dim; ++m) {
        for (int n = 0; n < dim - 1; ++n) {
            sum2 += sqrt(m * (n + 1)) * dm[m][n][m - 1][n + 1];
        }
    }
    // conj(a1) * a2
    for (int m = 0; m < dim - 1; ++m) {
        for (int n = 1; n < dim; ++n) {
            sum3 += sqrt((m + 1) * n) * dm[m][n][m + 1][n - 1];
        }
    }
    // conj(a1) * conj(a2)
    for (int m = 0; m < dim - 1; ++m) {
        for (int n = 0; n < dim - 1; ++n) {
            sum4 += sqrt((m + 1) * (n + 1)) * dm[m][n][m + 1][n + 1];
        }
    }
    return -0.25 * (sum1 - sum2 - sum3 + sum4);
}


//An average value of the square impulse quadrature for a specific channel.
//:param dm: Applied density matrix for 2 channels
//:param channel: Number of the channel.
//:return: Average value of the square impulse quadrature for a specific channel:
//<P^2> = (-1/4) * <(a - conj(a))^2>
template <size_t dim>
complex<double> impulse_square_aver(complex<double> (&dm)[dim][dim][dim][dim], int channel){
    complex<double> sum1, sum2, sum3;
    switch(channel) {
        case 1 :
            // -1 + 2*a*conj(a)
            for (int m = 0; m < dim; ++m) {
                for (int n = 0; n < dim; ++n) {
                    sum1 += (2.0*(m + 1) - 1) * dm[m][n][m][n];
                }
            }
            // a^2
            for (int m = 2; m < dim; ++m) {
                for (int n = 0; n < dim; ++n) {
                    sum2 =+ sqrt(m*(m - 1)) * dm[m][n][m - 2][n];
                }
            }
            // conj(a)^2
            for (int m = 0; m < dim - 2; ++m) {
                for (int n = 0; n < dim; ++n) {
                    sum3 += sqrt((m + 1)*(m + 2)) * dm[m][n][m + 2][n];
                }
            }
            break;
        case 2 :
            // -1 + 2*a*conj(a)
            for (int m = 0; m < dim; ++m) {
                for (int n = 0; n < dim; ++n) {
                    sum1 += (2.0*(n + 1) - 1) * dm[m][n][m][n];
                }
            }
            // a^2
            for (int m = 0; m < dim; ++m) {
                for (int n = 2; n < dim; ++n) {
                    sum2 += sqrt(n*(n - 1)) * dm[m][n][m][n - 2];
                }
            }
            // conj(a)^2
            for (int m = 0; m < dim; ++m) {
                for (int n = 0; n < dim - 2; ++n) {
                    sum3 += sqrt((n + 1)*(n + 2)) * dm[m][n][m][n + 2];
                }
            }
            break;
        default:
            throw invalid_argument("Invalid parameter");
    }
    return -0.25 * (-sum1 + sum2 + sum3);
}


//Average value of the square coordinate quadrature for a specific channel.
//:param dm: Applied density matrix in 2 channels
//:param channel: Number of the channel.
//:return: Average value of the square coordinate quadrature for a specific channel:
//<X^2> = (1/4) * <(a + conj(a))^2>
template <size_t dim>
complex<double> coord_square_aver(complex<double> (&dm)[dim][dim][dim][dim], int channel){
    complex<double> sum1, sum2, sum3;
    switch(channel) {
        case 1 :
            // -1 + 2*a*conj(a)
            for (int m = 0; m < dim; ++m) {
                for (int n = 0; n < dim; ++n) {
                    sum1 = sum1 + (2.0 * (m + 1) - 1) * dm[m][n][m][n];
                }
            }
            // a^2
            for (int m = 2; m < dim; ++m) {
                for (int n = 0; n < dim; ++n) {
                    sum2 += sqrt(m * (m - 1)) * dm[m][n][m - 2][n];
                }
            }
            // conj(a)^2
            for (int m = 0; m < dim - 2; ++m) {
                for (int n = 0; n < dim; ++n) {
                    sum3 += sqrt((m + 1) * (m + 2)) * dm[m][n][m + 2][n];
                }
            }
            break;
        case 2 :
            // -1 + 2*a*conj(a)
            for (int m = 0; m < dim; ++m) {
                for (int n = 0; n < dim; ++n) {
                    sum1 += (2.0 * (n + 1) - 1) * dm[m][n][m][n];
                }
            }
            // a^2
            for (int m = 0; m < dim; ++m) {
                for (int n = 2; n < dim; ++n) {
                    sum2 += sqrt(n * (n - 1)) * dm[m][n][m][n - 2];
                }
            }
            // conj(a)^2
            for (int m = 0; m < dim; ++m) {
                for (int n = 0; n < dim - 2; ++n) {
                    sum3 += sqrt((n + 1) * (n + 2)) * dm[m][n][m][n + 2];
                }
            }
            break;
        default :
            throw invalid_argument("Invalid parameter");
    }
    return 0.25 * (sum1 + sum2 + sum3);
}


// Two modes squeezing EPR operator variance.
// :param dm: Applied density matrix for 2 channels
// :return: EPR X operator's variance:
// var[EPR_X] = var[(X_2 - X_1)]
// Where var[A] is a variance of A: var[A] = <A^2> - (<A>)^2
template <size_t dim>
double epr_x_variance(complex<double> (&dm)[dim][dim][dim][dim]){
    complex<double> var = coord_square_aver(dm, 1) - 2.0 * prod_coord_aver(dm) + coord_square_aver(dm, 2) - pow(coord_aver(dm, 1), 2) + 2.0 * coord_aver(dm, 1) * coord_aver(dm, 2) - pow(coord_aver(dm, 2), 2);
    return real(var);
}


// Two modes squeezing EPR operator variance.
// :param dm: Applied density matrix for 2 channels
// :return: EPR P operator's variance:
// var[EPR_P] = var[(P_1 + P_2)]
// Where var[A] is a variance of A: var[A] = <A^2> - (<A>)^2
template <size_t dim>
double epr_p_variance(complex<double> (&dm)[dim][dim][dim][dim]){
    complex<double> var = impulse_square_aver(dm, 1) + 2.0 * prod_impulse_aver(dm) + impulse_square_aver(dm, 2) - pow(impulse_aver(dm, 1), 2) - 2.0 * impulse_aver(dm, 1) * impulse_aver(dm, 2) - pow(impulse_aver(dm, 2), 2);
    return real(var);
}


// Density matrix transformation at BS.
// a1 => t b2 + i r b1.
// a2 => t b1 + i r b2.
// Takes: Unapplied density matrix in 2 channels.
// Returns: Applied density matrix in 2 channels.
template <size_t dim>
void bs_transform_dm(
        double t,
        double r,
        complex<double> (&input_matrix)[dim][dim][dim][dim],
        complex<double> (&output_matrix)[dim*2][dim*2][dim*2][dim*2]
){
    int d1;
    int d2;
    int d1_;
    int d2_;
    complex<double> coeff1;
    complex<double> coeff2;
    complex<double> tc = t;
    complex<double> rc = r;

//    cout << "t: " << tc << endl;
//    cout << "r: " << rc << endl;

    for (int p1 = 0; p1 < dim; ++p1) {
        for (int p2 = 0; p2 < dim; ++p2) {
            for (int p1_ = 0; p1_ < dim; ++p1_) {
                for (int p2_ = 0; p2_ < dim; ++p2_) {

                    for (int n = 0; n < p1 + 1; ++n) {
                        for (int k = 0; k < p2 + 1; ++k) {
                            for (int n_ = 0; n_ < p1_ + 1; ++n_) {
                                for (int k_ = 0; k_ < p2_ + 1; ++k_) {
//                                    d2 = n + p2 - k;
//                                    d1 = p1 - n + k;
//                                    // TODO change channels and change it in EPR function.
//                                    //d1 = n + p2 - k;
//                                    //d2 = p1 - n + k;
//                                    coeff1 = pow(tc, p1 - n + p2 - k) * pow(I * rc, n + k) * fact_arr[p1] * fact_arr[p2] / (fact_arr[n] * fact_arr[p1 - n] * fact_arr[k] * fact_arr[p2 - k]);
//                                    d2_ = n_ + p2_ - k_;
//                                    d1_ = p1_ - n_ + k_;
//                                    // d1_ = n_ + p2_ - k_;
//                                    // d2_ = p1_ - n_ + k_;
//                                    coeff2 = pow(tc, p1_ - n_ + p2_ - k_) * pow(-I * rc, n_ + k_) * fact_arr[p1_] * fact_arr[p2_] / (fact_arr[n_] * fact_arr[p1_ - n_] * fact_arr[k_] * fact_arr[p2_ - k_]);
//                                    // unapplied.
//                                    //output_matrix[d1][d2][d1_][d2_] += input_matrix[p1][p2][p1_][p2_] * coeff1 * coeff2;
//
//                                    // applied.
//                                    output_matrix[d1][d2][d1_][d2_] += input_matrix[p1][p2][p1_][p2_] * coeff1 * coeff2 * fact_sqrt_arr[d1]*fact_sqrt_arr[d2]*fact_sqrt_arr[d1_]*fact_sqrt_arr[d2_];

                                    d1 = p1 - n + k;
                                    d2 = n + p2 - k;
                                    coeff1 = pow(tc, p1 - n + p2 - k) * pow(I * rc, n + k) * fact_sqrt_arr[d1] * fact_sqrt_arr[d2] * fact_sqrt_arr[p1] * fact_sqrt_arr[p2] / (fact_arr[n] * fact_arr[p1 - n] * fact_arr[k] * fact_arr[p2 - k]);
                                    d1_ = p1_ - n_ + k_;
                                    d2_ = n_ + p2_ - k_;
                                    coeff2 = pow(tc, p1_ - n_ + p2_ - k_) * pow(-I * rc, n_ + k_) * fact_sqrt_arr[d1_] * fact_sqrt_arr[d2_] * fact_sqrt_arr[p1_] * fact_sqrt_arr[p2_] / (fact_arr[n_] * fact_arr[p1_ - n_] * fact_arr[k_] * fact_arr[p2_ - k_]);
                                    // unapplied.
                                    //output_matrix[d1][d2][d1_][d2_] += input_matrix[p1][p2][p1_][p2_] * coeff1 * coeff2;

                                    // applied.
                                    output_matrix[d1][d2][d1_][d2_] += input_matrix[p1][p2][p1_][p2_] * coeff1 * coeff2;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


template <size_t dim_in, size_t dim_out>
void make_dm_appl(
        complex<double> (&dm_in)[dim_in][dim_in][dim_in][dim_in],
        complex<double> (&dm_out)[dim_out][dim_out][dim_out][dim_out]
                ){
    for (int p1 = 0; p1 < dim_out; ++p1) {
        for (int p2 = 0; p2 < dim_out; ++p2) {
            for (int p1_ = 0; p1_ < dim_out; ++p1_) {
                for (int p2_ = 2; p2_ < dim_out; ++p2_) {
                    dm_out[p1][p2][p1_][p2_] = dm_in[p1][p2][p1_][p2_] * sqrt(fact_arr[p1] * fact_arr[p2] * fact_arr[p1_] * fact_arr[p2_]);
                }
            }
        }
    }
}


template <size_t st_dim, size_t d>
void evaluate_system(
        complex<double> input_state[st_dim][st_dim],
        map<string, float> bs_params,
        double phase_diff,
        int phase_mod_channel,
        char det_event,
        complex<double> (&dm_out)[d][d][d][d],
        double &prob
){
    // First BS transformation
    static complex<double> state1[2*st_dim][2*st_dim];
    memset(state1, 0, sizeof state1);
    bs2x2_transform(bs_params["t1"], bs_params["r1"], input_state, state1);

    // 2nd and 3rd BS transformations.
    static complex<double> state2[2*st_dim][2*st_dim][2*st_dim][2*st_dim];
    memset(state2, 0, sizeof state2);
    two_bs2x4_transform(bs_params["t2"], bs_params["r2"], bs_params["t3"], bs_params["r3"], state1, state2);

    // Detection probability.
    prob = det_probability(det_event, state2);

    // The detection event. Gives non-normalised state.
    static complex<double> state_proj[2*st_dim][2*st_dim][2*st_dim][2*st_dim];
    memset(state_proj, 0, sizeof state_proj);
    detection(det_event, state2, state_proj);

    static double norm_after_det;
    norm_after_det = state_norm(state_proj);
    //cout << "Norm aft detection." << norm_after_det << "\n";

    // Renormalizing the state.
    static complex<double> state_proj_norm[2*st_dim][2*st_dim][2*st_dim][2*st_dim];
    memset(state_proj_norm, 0, sizeof state_proj_norm);
    renormalize_state(norm_after_det, state_proj, state_proj_norm);

    static complex<double> dm1[dim_dm][dim_dm][dim_dm][dim_dm];
    memset(dm1, 0, sizeof dm1);
    build_dm_with_trace(state_proj_norm, dm1);

    // Phase modulation.
    static complex<double> dm2[dim_dm][dim_dm][dim_dm][dim_dm];
    memset(dm2, 0, sizeof dm2);
    phase_modulation(phase_mod_channel, phase_diff, dm1, dm2);

    bs_transform_dm(bs_params["t4"], bs_params["r4"], dm2, dm_out);
}


// ./main 'F' 0.0
int main(int argc, char** argv) {
    // Initialize utils.
    for (int i = 0; i < sz; ++i) {
        fact_arr[i] = tgamma(i + 1);
        fact_sqrt_arr[i] = sqrt(tgamma(i + 1));
    }

    //
    // System marameters.
    //
    // Detection event.
    // 'F' - first. 'T' - third. 'N' - none. 'B' - both.
    //char det = 'F';
    char det = *argv[1];

    const int phase_mod_channel = 1;

    //double phase_in_phi = 0.0;
    double phase_in_phi = atof(argv[2]);
    const double phase = M_PI * phase_in_phi;

    //const double prob_constraint = 0.1;

    const size_t r1_grid = 17;
    const size_t r4_grid = 17;
    const size_t r2_grid = 17;
    const size_t r3_grid = 17;

    const double min_bound = 1e-5;
    const double max_bound = 1.0 - 1e-5;

    // BS values range.
    double T1_min = 0.5;
    double T1_max = 1.0;
    double T4_min = 0.5;
    double T4_max = 1.0;

    double T2_min = min_bound;
    double T2_max = max_bound;
    double T3_min = min_bound;
    double T3_max = max_bound;

    // BS parameters arrays.
    double t1_arr [r1_grid];
    double r1_arr [r1_grid];
    double t2_arr [r2_grid];
    double r2_arr [r2_grid];
    double t3_arr [r3_grid];
    double r3_arr [r3_grid];
    double t4_arr [r4_grid];
    double r4_arr [r4_grid];

    build_bs_parameters(T1_min, T1_max, t1_arr, r1_arr);
    build_bs_parameters(T2_min, T2_max, t2_arr, r2_arr);
    build_bs_parameters(T3_min, T3_max, t3_arr, r3_arr);
    build_bs_parameters(T4_min, T4_max, t4_arr, r4_arr);

    //
    // Input states
    //
    // Bottom channel.
    static complex<double> state_ch1[L];
    // Top channel.
    static complex<double> state_ch2[L];

    build_coherent_state(1.2, state_ch2);
    build_fock_state(1, state_ch1);

    static complex<double> input_state[L][L];
    for (size_t p1 = 0; p1 < L; ++p1) {
        for (size_t p2 = 0; p2 < L; ++p2) {
            input_state[p1][p2] = state_ch1[p1] * state_ch2[p2];
        }
    }

    // Output variables.
    // static double epr_x_arr[r1_grid][r2_grid][r3_grid][r4_grid];
    // static double epr_p_arr[r1_grid][r2_grid][r3_grid][r4_grid];
    // static double prob_arr[r1_grid][r2_grid][r3_grid][r4_grid];

    static double epr_x_min = 100.0;
    // double epr_p_min = 100.0;

    static double prob_min;

    cout << "DET: " << det << "\n";
    cout << "Phase: " << phase_in_phi << "\n";

    string save_root = "/home/matvei/qscheme/cpp_optimized/results/";
    //string fname = "coh_single_DET-F.txt";
    string fname = "coh_single_DET-F_full.txt";
    string save_path = save_root + fname;
    cout << "Saving path: " << save_path << "\n";

    // Start evaluation.
    auto start = std::chrono::high_resolution_clock::now();

    ofstream f;
    f.open(save_path, ios::app);

    for (size_t n1 = 0; n1 < r1_grid; ++n1) {
        cout << "Step: " << n1 << "\n";
        for (size_t n4 = 0; n4 < r4_grid; ++n4) {
            for (size_t n2 = 0; n2 < r2_grid; ++n2) {
                for (size_t n3 = 0; n3 < r3_grid; ++n3) {

                    map<string, float> bs_params = {
                            {"t1", t1_arr[n1]},
                            {"r1", r1_arr[n1]},
                            {"t2", t2_arr[n2]},
                            {"r2", r2_arr[n2]},
                            {"t3", t3_arr[n3]},
                            {"r3", r3_arr[n3]},
                            {"t4", t4_arr[n4]},
                            {"r4", r4_arr[n4]},
                    };

                    // Evaluate the whole system.
                    static complex<double> dm_out[dim_dm*2][dim_dm*2][dim_dm*2][dim_dm*2];
                    memset(dm_out, 0, sizeof dm_out);
                    double prob;
                    evaluate_system(input_state, bs_params, phase, phase_mod_channel, det, dm_out, prob);
                    //cout << "Prob: " << prob << "\n";

                    double epr_x = epr_x_variance(dm_out);
                    //cout << "EPR X: " << epr_x << " Prob: " << prob << "\n";

                    f << epr_x << ',' << prob << endl;

                    //epr_x_arr[n1][n2][n3][n4] = epr_x;
                    //prob_arr[n1][n2][n3][n4] = prob;

//                    static complex<double> sum2;
//                    sum2 = 0.0;
//                    for (int p = 0; p < dim_dm*2; ++p) {
//                        sum2 += dm_out[p][p][p][p];
//                    }
//                    cout << "Sum: " << sum2 << endl;

//                    if (epr_x < epr_x_min and prob > prob_constraint) {
//                        epr_x_min = epr_x;
//                        prob_min = prob;
//                    }
                }
            }
        }
    }

    cout << "EPR X min: "<< epr_x_min << "\n";
    cout << "Minimizing probability: "<< prob_min << "\n";

//    ofstream f;
//    f.open(save_path, ios::app);
//    f << phase_in_phi << ',' << epr_x_min << ',' << prob_min << endl;
//    f.close();

    f.close();

    auto finish = chrono::high_resolution_clock::now();
    show_time(start, finish);
    return 0;
}

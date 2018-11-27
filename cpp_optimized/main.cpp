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


int main() {
    auto start = std::chrono::high_resolution_clock::now();

    char DET = 'F';

    for (int i = 0; i < sz*4; ++i) {
        fact_arr[i] = factorial(i);
        fact_sqrt_arr[i] = sqrt(factorial(i));
    }

    //dm = evaluate_system();

    auto finish = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = finish - start;
    cout << "Elapsed time: " << elapsed.count() << " s\n";
    cout << "shit" << endl;
    return 0;
}


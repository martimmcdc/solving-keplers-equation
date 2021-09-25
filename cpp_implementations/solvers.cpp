#include <iostream>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double pi = M_PI;

// Newton-Raphson solver
double newton_solver(double e, double M, double epsilon=1.0e-9){

    double x = M + 0.85*e; // initial guess
    double h0 = x - e*sin(x) - M; // function value
    double h1;
    while (abs(h0) > epsilon){
            h1 = 1 - e*cos(x);     // first derivative
            x -= h0/h1;            // newton-raphson correction
            h0 = x - e*sin(x) - M; // function
        }
    return x;
}


// Iterative solver
double iterative_solver(double e, double M, double epsilon=1.0e-9){

    double x = M + 0.85*e; // initial guess
    double h0 = x - e*sin(x) - M; // function value
    while (abs(h0) > epsilon){
        x -= h0; // x --> e*sin(x) + M = x - h0
        h0 = x - e*sin(x) - M; // function value
    } 
    return x;
}

// Kepler's Goat Herd solver
double goat_herd_solver(double e, double M, int N=10){
    
    double x0 = M - 0.5*e; // contour center
    double dx = 0.5*e;     // contour radius

    double x[N]; // integration variable array
    x[0] = 0;
    double delta = 0.5/(N - 1);
    for(int i = 1; i < N; i++){
        x[i] = x[i-1] + delta;
    }

    if (M < pi){
        x0 += e;
    }

    // Pre calculation of trigonometric and hyperbolic functions
    double cosx0 = cos(x0);
    double sinx0 = sin(x0);

    double arg[N];
    double cos2pix[N]; // real part of exp(-2i.pi.k.x) , k = -1
    double sin2pix[N]; // imag part of exp(-2i.pi.k.x) , k = -1
    double cos4pix[N]; // real part of exp(-2i.pi.k.x) , k = -2
    double sin4pix[N]; // imag part of exp(-2i.pi.k.x) , k = -2
    double dxcos[N]; // argument of trigonometric functions that appear in expansion of e*sin(x) below 
    double dxsin[N]; // argument of hyperbolic functions that appear in expansion of e*sin(x) below
    double ecosdx[N];
    double esindx[N];
    double coshdx[N];
    double sinhdx[N];

    // f(x) = x - e*sin(x) - M
    double fR[N]; // real part of f(x)
    double fI[N]; // imag part of f(x)

    // Integrand functions up to constant which disappears in division
    // Re[d(ak)/dx] = Re[a(x)*exp(-2i.pi.k.x)], where a(x) = 1/f(x) = 1/(fR + ifI)
    // such that: aR = fR/(fR^2 + fI^2) , aI = -fI/(fR^2 + fI^2)
    double a_1sub[N];
    double a_2sub[N];
    double abs_f[N];

    double a_1 = 0;
    double a_2 = 0;

    // Evaluate arrays
    for(int i = 0; i < N; i++){
        arg[i] = 2*pi*x[i];
        cos2pix[i] = cos(arg[i]);
        sin2pix[i] = sin(arg[i]);
        cos4pix[i] = cos2pix[i]*cos2pix[i] - sin2pix[i]*sin2pix[i];
        sin4pix[i] = 2*sin2pix[i]*cos2pix[i];
        dxcos[i] = dx*cos2pix[i];
        dxsin[i] = dx*sin2pix[i];
        ecosdx[i] = e*cos(dxcos[i]);
        esindx[i] = e*sin(dxcos[i]);
        coshdx[i] = cosh(dxsin[i]);
        sinhdx[i] = sinh(dxsin[i]);

        fR[i] = x0 + dxcos[i] - (sinx0*ecosdx[i] + cosx0*esindx[i])*coshdx[i] - M;
        fI[i] = dxsin[i] - (cosx0*ecosdx[i] - sinx0*esindx[i])*sinhdx[i];

        a_1sub[i] = fR[i]*cos2pix[i] + fI[i]*sin2pix[i];
        a_2sub[i] = fR[i]*cos4pix[i] + fI[i]*sin4pix[i];
        abs_f[i] = (fR[i]*fR[i] + fI[i]*fI[i]);
        a_1sub[i] /= abs_f[i];
        a_2sub[i] /= abs_f[i];
        a_1 += a_1sub[i];
        a_2 += a_2sub[i];
    }

    // Trapezoidal integral correction
    a_1 -= 0.5*(a_1sub[0] + a_1sub[N-1]);
    a_2 -= 0.5*(a_2sub[0] + a_2sub[N-1]);
    return x0 + dx*a_2/a_1;
}

// Danby solver
double danby_solver(double e, double M, double epsilon=1e-9){

    double x = M + 0.85*e; // initial guess suggested by Danby III
    double h2 = e*sin(x);
    double h0 = x - h2 - M; // function value
    double h1,h3,d1,o2,d2,d3;

    while(abs(h0) > epsilon){
        h3 = e*cos(x);           // 3rd derivative
        h1 = 1-h3;                  // 1st derivative
        d1 = -h0/h1;                // 1st order correction
        o2 = h1 + 0.5*d1*h2;        // denominator for 2nd order
        d2 = -h0/o2;                // second order
        d3 = -h0/(o2 + d2*d2*h3/6); // 3rd order correction
        x += d3;                    // update
        h2 = e*sin(x);           // 2nd derivative
        h0 = x - h2 - M;            // 0th derivative
    }
    return x;
}

// Factorial
int factorial(int num){
    int fact = 1;
    for(int i = 2; i < num+1; i++){
        fact *= i;
    }
    return fact;
}

// Nijenhuis solver
double nijenhuis_method(double e, double M, int order=1){
    
    double e_1 = 1 - e;
    double mikkoleft = 0.45;
    double mikkolup = 1 - mikkoleft;
    double E_rgh; // rough starter
    double E_ref; // refined starter

    // rough starters and refinement through Halley's method in A, B and C and Newton-Raphson in D
    double h0,h1;

    if(e > mikkolup && M < mikkoleft){
        double denom = 4 * e + 0.5;
        double q = M/(2 * denom);
        double p = e_1/denom;
        double p2 = p * p;
        double z2 = pow(sqrt(p * p2 + q * q) + q, 2/3);
        E_rgh = 2 * q/(z2 + p + p2/z2);
        double s2 = E_rgh * E_rgh;
        double s4 = s2 * s2;
        h0 = 0.075 * s4 * E_rgh;
        h1 = 0.375 * s4 + denom * s2 + e_1;
        E_rgh -= h0/h1;
        E_ref = M + e * E_rgh * (3 - 4 * s2); // refined starter in region D
    }else{

        // regions
        double lim1 = pi - 1 - e;
        double lim2;
        if(e <= mikkolup){
            lim2 = e_1;
        }else{
            lim2 = mikkoleft;
        }

        if(M >= lim1){
            E_rgh = (M + pi * e)/(1 + e); // rough starter in region A
        }else if(M < lim1 && M >= lim2){
            E_rgh = M + e; // rough starter in region B
        }else{
            E_rgh = M/e_1; // rough starter in region C
        }
        double esinx = e*sin(E_rgh);
        double ecosx = e*cos(E_rgh);
        h0 = E_rgh - esinx - M;
        h1 = 1 - ecosx;
        E_ref = E_rgh - h0 * h1/(h1 * h1 - 0.5 * h0 * esinx); // refined starter in regions A, B and C
    }

    // Final Step

    // Pre-compute e*sin(E) and e*cos(E)
    double esinE = e*sin(E_ref);
    double ecosE = e*cos(E_ref);

    // Arrays to store values in terms of order (1st index) and different M values (2nd index)
    double func[order+1];  // f and its m = order derivatives
    double h[order+1];     // h constants (0th order has no meaning)
    double delta[order+1]; // delta constants (0th order has no meaning)
    int f;

    // Substitute function's and derivatives' values into array
    for(int i = 0; i < order + 1; i += 4){
        func[i]  = -esinE; // 0th, 4th, ... order derivatives
        func[i+1]= -ecosE; // 1st, 5th, ... order derivatives
        func[i+2]= esinE;  // 2nd, 6th, ... order derivatives
        func[i+3]= ecosE;  // 3rd, 7th, ... order derivatives
    }

    func[0] += E_ref - M; // add linear and constant terms to function
    func[1] += 1;         // add linear term's 1st derivative

    delta[1] = func[1];       // 1st delta value
    h[1] = -func[0]/delta[1]; // 1st h value

    // Compute h[i] from i=2 to i=order, using h[j] values from j=1 to j=i-1
    std::cout << order << std::endl;
    for(int i = 2; i < order + 1; i++){
        for(int j = 1; j < i; j++){
            f = factorial(i-j+1);
            delta[i] += func[i-j+1]/f;
            delta[i] *= h[j];
            delta[i] += func[1];
            // std::cout << "i = " << i << ", j = " << j << "i - j + 1 = " << i-j+1 << std::endl;
            // std::cout << "f = " << f << ", delta[i] = " << delta[i] << "\n" << std::endl;
            h[i] = -func[0]/delta[i];
        }
    }
    return E_ref + h[order];
}

// Nijenhuis' solver
double nijenhuis_solver(double e, double M, double epsilon=1e-9){
    double x = M;
    double h0 = x - e*sin(x) - M;
    int m = 0;
    while(abs(h0) > epsilon){
            m += 1; // raise order
            x = nijenhuis_method(e,M,m);
            h0 = x - e*sin(x) - M;
        }
    return x;
}


double cordic_solver(double e, double M, int n = 29){

    double pi2 = 2*pi;
    double E = pi2*floor(M/pi2 + 0.5); // initial values for E
    double cosE = 1.; // initial value for cos(E)
    double sinE = 0.; // initial value for sin(E)
    double newCosE;
    double newSinE;
    double sigma;
    double a;// angle: a
    double c,s; // cos(a), sin(a)

    for(int i = 0; i < n; i++){

        a = pi2/pow(2,(i+2));
        c = cos(a);
        s = sin(a);

        if(E > M + e*sinE){
            sigma = -1;
        }else{
            sigma = 1;
        }

        s *= sigma;
        E += a*sigma;
        newCosE = cosE*c-sinE*s;
        newSinE = cosE*s+sinE*c;
        cosE = newCosE;
        sinE = newSinE;
    }
    return E;
}



double KeplerStart3(double e, double M){

    double t34 = e*e;
    double t35 = e*t34;
    double t33 = cos(M);
    return M+(-0.5 * t35 + e + (t34 + 1.5 * t33 * t35) * t33) * sin(M);
}

double eps3(double e, double M, double x){

    double t1 = cos(x);
    double t2 = -1 + e * t1;
    double t3 = sin(x);
    double t4 = e * t3;
    double t5 = -x + t4 + M;
    double t6 = t5/(0.5 * t5 * t4/t2 + t2);
    return t5/((0.5 * t3 - 1/6 * t1 * t6) * e * t6 + t2);
}

double murison_solver(double e, double M, double epsilon=1e-9){

    double Mnorm = fmod(M,2*pi);
    double E = KeplerStart3(e,Mnorm);
    double h0 = E - e*sin(E) - M;

    while(abs(h0) > epsilon){
        E -= eps3(e,Mnorm,E);
        h0 = E - e*sin(E) - M;
    }
    return E;
}


PYBIND11_MODULE(cpp_solvers, module){
    module.doc() = "This module is a C++ implementation of algorithms to solve Keplers equation";
    module.def("newton_solver", py::vectorize(&newton_solver), "Newton-Raphson solver for Kepler's equation",
        py::arg("e"),py::arg("M"),py::arg("epsilon")=1e-9);
    module.def("iterative_solver", py::vectorize(&iterative_solver), "Iterative solver for Kepler's equation",
        py::arg("e"),py::arg("M"),py::arg("epsilon")=1e-9);
    module.def("goat_herd_solver", py::vectorize(&goat_herd_solver), "Contour integral solver for Kepler's equation",
        py::arg("e"),py::arg("M"),py::arg("N")=10);
    module.def("danby_solver", py::vectorize(&danby_solver), "Danby's solver for Kepler's equation",
        py::arg("e"),py::arg("M"),py::arg("epsilon")=1e-9);
    module.def("nijenhuis_solver", py::vectorize(&nijenhuis_solver), "Nijenhuis' solver for Kepler's equation",
        py::arg("e"),py::arg("M"),py::arg("epsilon")=1e-2);
    module.def("cordic_solver", py::vectorize(&cordic_solver), "CORDIC solver for Kepler's equation",
        py::arg("e"),py::arg("M"),py::arg("n")=29);
    module.def("murison_solver", py::vectorize(&murison_solver), "Murison's solver for Kepler's equation",
        py::arg("e"),py::arg("M"),py::arg("epsilon")=1e-9);
}

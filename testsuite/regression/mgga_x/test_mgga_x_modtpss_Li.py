
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_modtpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_modtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.960201833332258e+00, -1.361624618758485e+00, -4.064201361581152e-01, -1.760997945442837e-01, -7.740957300583268e-02, -2.054607474622543e-02, -3.838587222523154e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_modtpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_modtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.398843220319075e+00, -2.401013783755071e+00, -1.694550213078383e+00, -1.696280863489558e+00, -3.539835770952858e-01, -3.547237901230952e-01, -2.134767960550144e-01, -2.612127983609464e-02, -7.266593711738258e-02, -8.296437722073323e-04, -2.746373391122341e-02, -2.726608788142596e-02, -5.541556434923157e-04, -3.939542509439069e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_modtpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_modtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-9.749981458932273e-04, 0.000000000000000e+00, -9.741394546674616e-04, -1.602670087393116e-03, 0.000000000000000e+00, -1.599933891589304e-03, -9.245161907118007e-02, 0.000000000000000e+00, -9.179252621367778e-02, -2.730565608072468e+01, 0.000000000000000e+00, -2.422948329640078e-01, -7.020122832205382e+01, 0.000000000000000e+00, -1.547458388542185e+00, -2.462623394489016e-01, 0.000000000000000e+00, -2.299511353474991e-01, -1.126489857841555e+00, 0.000000000000000e+00, -1.612453391353764e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_modtpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_modtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([5.054578166196569e-02, 5.067165623862436e-02, 2.691782203574536e-02, 2.699981583174334e-02, 5.047774914586503e-04, 4.284404452509158e-04, 9.879044535740663e-01, 5.718191775165312e-11, 1.284746996849795e-02, 3.088792692634559e-17, 3.584806818424455e-17, 6.554609955264217e-11, 5.872117239065761e-38, 5.142531109154695e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

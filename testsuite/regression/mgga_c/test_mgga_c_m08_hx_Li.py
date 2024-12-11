
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m08_hx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.418245736167636e-01, -6.459429179231520e-02, -2.372331373617051e-01, -2.659625753818623e-02, -4.221084099441874e-02, -4.716355061333927e-02, -1.170169207492127e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m08_hx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.895367096438454e-01, -5.892910080002951e-01, -4.574472432931383e-02, -4.563769753688342e-02, 2.012493789267630e-02, 1.991672553965351e-02, -1.128017805269013e-01, -2.646996332164540e-01, 2.851414017814797e-03, -2.221732193231304e-01, -5.928619710944097e-02, -5.995142989154484e-02, -1.376583649954237e-03, -2.019943645919528e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_hx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.115972191557510e-04, 1.023194438311502e-03, 5.115972191557510e-04, -7.952137517219950e-04, -1.590427503443990e-03, -7.952137517219950e-04, 8.573094250960481e-01, 1.714618850192096e+00, 8.573094250960481e-01, 1.336872588401963e+01, 2.673745176803925e+01, 1.336872588401963e+01, 7.263108995295827e+02, 1.452621799059165e+03, 7.263108995295827e+02, -2.753673676899243e-07, -5.507347502879926e-07, -2.753673676899243e-07, -2.901544450800406e-15, 5.132542952782754e-14, -2.901544450800406e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_hx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([6.167569875446693e-02, 6.167569875446699e-02, -3.753225061582240e-03, -3.753225061582243e-03, -5.354282184749287e-02, -5.354282184749423e-02, 2.223448759833623e+00, 2.223448759833140e+00, -4.464998225351587e-01, -4.464998222273808e-01, -3.848220989178346e-08, -3.848220975267089e-08, -9.657231085139439e-20, -9.667570732982198e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

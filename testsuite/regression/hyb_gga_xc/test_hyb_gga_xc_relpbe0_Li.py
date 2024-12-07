
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_relpbe0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_relpbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.387204717537386e+00, -9.739246491525138e-01, -2.684240663182943e-01, -1.309516994612745e-01, -5.456073244750757e-02, -1.900509250665072e-02, -3.654759277742336e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_relpbe0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_relpbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.835203661731714e+00, -1.836732235229093e+00, -1.279175377266398e+00, -1.280163879360029e+00, -2.836435618523687e-01, -2.834842825855739e-01, -1.743553277377613e-01, -8.728026372557181e-02, -6.488529592198655e-02, 1.151247514580701e-01, -2.509267346685594e-02, -2.495986227101641e-02, -5.177727633683482e-04, -4.008598970130059e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_relpbe0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_relpbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.136943121375377e-05, 3.608203618403554e-05, -2.122863356176822e-05, -1.090451666505923e-04, 1.167195095087809e-04, -1.084814866264208e-04, -3.600652912675614e-02, 2.384112668150253e-03, -3.602675627454126e-02, 7.466263184199685e-01, 2.677665271000139e+00, -4.353614394398361e-01, -1.616794437923757e+01, 8.651582382802433e+00, -7.184322947882255e+00, -1.799984290397336e+00, 1.265610119299689e-04, -1.682121953088340e+00, -8.379125514460162e+00, 1.211170118747835e-06, -1.199392152799071e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

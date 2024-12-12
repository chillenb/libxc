
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_mn12_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.381830861137152e-01, -1.314678899076535e-01, 1.202148828099125e-01, -4.458207835116688e-02, 2.216035770366559e-02, -4.808811793030565e-02, -1.193367781549517e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_mn12_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.981041126787795e-01, -5.976964791451375e-01, -3.051868847070989e-01, -3.049832427013221e-01, 1.095310914803131e-01, 1.096075427002532e-01, -1.184658308098650e-01, -3.761040271152030e-01, -1.499886421482646e-02, 3.123606998232101e+00, -6.043123661420079e-02, -6.110951148099203e-02, -1.403874384884275e-03, -2.059988976012188e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.016569160827862e-05, -2.033138321655723e-05, -1.016569160827862e-05, -9.407206605017117e-05, -1.881441321003423e-04, -9.407206605017117e-05, 3.071974929338595e-02, 6.143949858677192e-02, 3.071974929338595e-02, -7.744779433944174e-01, -1.548955886788835e+00, -7.744779433944174e-01, 9.566139796411609e+01, 1.913227959282321e+02, 9.566139796411609e+01, 1.480276284771392e-03, 2.960552569609595e-03, 1.480276284771392e-03, 1.416741666246639e-05, 2.833487821528687e-05, 1.416741666246639e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([4.945109474832525e-02, 4.945109474832523e-02, 2.830822927041510e-02, 2.830822927041509e-02, -1.733431565114312e-02, -1.733431565114312e-02, 1.850866684522038e+00, 1.850866684521634e+00, -5.021042500069326e-03, -5.021042496608331e-03, -1.766799592708170e-07, -1.766799592700924e-07, -4.668867983397683e-19, -4.668867983397682e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

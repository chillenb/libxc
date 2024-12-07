
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbefe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbefe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.821207400434294e+00, -1.312280568891243e+00, -3.875235366456694e-01, -1.616004916209878e-01, -7.922717629405851e-02, -1.637252327396429e-02, -3.057678252181200e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbefe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbefe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.225819227395688e+00, -2.227926055518672e+00, -1.521803694647598e+00, -1.523124038433088e+00, -4.580723754049711e-01, -4.581138034956739e-01, -2.038592379046683e-01, -2.082925305687694e-02, -8.865288171579504e-02, -6.608654666678594e-04, -2.190239276580963e-02, -2.174354684684284e-02, -4.414204399254352e-04, -3.138096499315419e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbefe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbefe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.452122846666453e-04, 0.000000000000000e+00, -3.441115519225516e-04, -1.197095443635156e-03, 0.000000000000000e+00, -1.193596615407322e-03, -2.846322899280244e-02, 0.000000000000000e+00, -2.834262017943761e-02, -5.679791772217428e+00, 0.000000000000000e+00, -5.213716104399790e-02, -3.716357035385874e+01, 0.000000000000000e+00, -3.329616897743317e-01, -5.299235939101517e-02, 0.000000000000000e+00, -4.948123698868591e-02, -2.423832133400367e-01, 0.000000000000000e+00, -3.469464163999615e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

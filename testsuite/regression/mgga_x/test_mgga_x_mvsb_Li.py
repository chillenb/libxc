
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mvsb_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.038110198516533e+00, -1.414830045461314e+00, -3.236643513022445e-01, -1.844508777817815e-01, -7.214900369517691e-02, -2.745564255967904e-03, -8.206942670914908e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mvsb_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.826473966619408e+00, -2.828745230980235e+00, -2.130450981542476e+00, -2.131384366583255e+00, -4.453919275172358e-01, -4.436204164015710e-01, -2.506537131722918e-01, -5.134101828955708e-03, -8.577704306726310e-02, -3.655839073766213e-05, -5.109060362677986e-03, -5.440379054354206e-03, -1.764533267827056e-05, -1.202402499990603e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvsb_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.017623738431907e-04, 0.000000000000000e+00, 4.993835542108477e-04, 3.432706593493690e-03, 0.000000000000000e+00, 3.402528118323622e-03, -3.411070269771140e-02, 0.000000000000000e+00, -3.950559730787154e-02, 6.667555721694601e+00, 0.000000000000000e+00, 7.225919899845833e+00, -7.730022218563623e+01, 0.000000000000000e+00, 4.075392186752732e+03, 1.407328736112532e+01, 0.000000000000000e+00, 6.557017331256243e+00, 1.280705792413629e+04, 0.000000000000000e+00, 1.277931988296441e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvsb_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.596900699388252e-02, -2.591112454305422e-02, -5.949607244904741e-02, -5.903244606461772e-02, 1.544292748796827e-02, 1.728978712775978e-02, -2.578720762728463e-01, 1.344453334218985e-04, 2.085184513615123e-01, 2.417742224419904e-06, 3.773129417752571e-08, 1.387952066021707e-04, 3.999996796229500e-17, 8.117195246234048e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

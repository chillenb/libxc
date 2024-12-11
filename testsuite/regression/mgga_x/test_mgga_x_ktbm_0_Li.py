
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.879965759719377e+00, -1.259119004864442e+00, -2.951181460314590e-01, -1.725459388140028e-01, -6.075688820406638e-02, -1.318468755840339e-02, -2.438528296959644e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.626611434332615e+00, -2.629169708996782e+00, -1.811744144821934e+00, -1.813401989870450e+00, -3.646291678965167e-01, -3.637088801126348e-01, -2.373515633931784e-01, -1.638541502943910e-02, -7.641859275884248e-02, -5.196768935057242e-04, -1.723060855933522e-02, -1.710504557726964e-02, -3.471143240242678e-04, -2.408706758955657e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.210975731436071e-04, 0.000000000000000e+00, -5.193090325586090e-04, -2.038783030623809e-03, 0.000000000000000e+00, -2.032884281647353e-03, -6.805921675983362e-02, 0.000000000000000e+00, -7.056778631044697e-02, -8.039504068504659e+00, 0.000000000000000e+00, -1.008898397647273e+01, -8.683852816255410e+01, 0.000000000000000e+00, -2.519683068929284e+04, -1.870872022318140e-01, 0.000000000000000e+00, -9.022337643408989e+00, -3.815492041743157e-01, 0.000000000000000e+00, -2.304465874514798e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.728457226556758e-02, 1.727381475154757e-02, 1.955561849556358e-02, 1.956566077410962e-02, -5.310253901833638e-03, -5.479625497368179e-03, 2.053863284273668e-01, 1.290662115014003e-04, -3.843699452125288e-02, 1.026612448047202e-05, 2.778569644184977e-06, 1.313195983856130e-04, 4.632619890788233e-11, -1.267111875192943e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

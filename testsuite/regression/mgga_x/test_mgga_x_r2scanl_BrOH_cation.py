
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_r2scanl_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.219373717365978e+01, -2.219375093061895e+01, -2.219390832416882e+01, -2.219370208180157e+01, -2.273241370785216e+01, -2.273241370785216e+01, -3.744087110696520e+00, -3.788781067767232e+00, -3.789062922660022e+00, -3.785737184348518e+00, -3.744286688516419e+00, -3.788874291619534e+00, -7.307863017195207e-01, -7.299834931918350e-01, -7.106039821003214e-01, -7.164869654195771e-01, -7.136934102209059e-01, -7.167543078074442e-01, -1.848500162832601e-01, -1.879532094098479e-01, -8.509985322974796e-01, -1.432675711399805e-01, -1.812245764800494e-01, -1.802812159884666e-01, -4.921728167439221e-03, -3.321984451012152e-03, -3.820075942861595e-02, -2.578790886032041e-03, -4.988350507248696e-03, -2.494563611590809e-03, -5.672600085168801e+00, -5.481008568636131e+00, -5.672651959349361e+00, -5.673571272615735e+00, -5.673125868599428e+00, -5.481492039889289e+00, -2.125212210997748e+00, -2.140832265646797e+00, -2.120930661127320e+00, -2.127240903475222e+00, -2.146247264612791e+00, -2.139643915575011e+00, -6.392432182789721e-01, -6.487893916124561e-01, -5.918758968438186e-01, -5.845178437910395e-01, -6.503928347033312e-01, -6.503928347033312e-01, -9.176542370619045e-02, -1.889846202731029e-01, -9.213775200114224e-02, -1.865625161482534e+00, -1.169116576358870e-01, -1.136425233979884e-01, -1.920280269361845e-03, -3.709028205575127e-03, -2.077381008328198e-03, -6.188688348471526e-02, -3.568059111878118e-03, -3.568059111878119e-03, -6.015704325428188e-01, -6.046398885771302e-01, -6.037073130929187e-01, -6.410294316270490e-01, -6.033388438751923e-01, -6.404949195186695e-01, -6.249364186342161e-01, -5.364649909212400e-01, -5.617648073266222e-01, -5.868560088335203e-01, -5.633090943250676e-01, -5.740407204582494e-01, -7.280308747409774e-01, -2.360350961384786e-01, -2.867615289062423e-01, -3.778534238287382e-01, -3.293791755501025e-01, -3.293791755501025e-01, -4.929134777689930e-01, -3.143486279497305e-02, -4.580363684953172e-02, -3.640676584752027e-01, -7.891630680087461e-02, -7.891630680087461e-02, -6.606168965914755e-03, -9.913377696918121e-04, -1.320965001508062e-03, -7.026584496094537e-02, -3.276646272115884e-03, -2.792317576567403e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08

# test_mgga_x_r2scanl_BrOH_cation_2_vrho() not generated due to NaN

# test_mgga_x_r2scanl_BrOH_cation_2_vsigma() not generated due to NaN

# test_mgga_x_r2scanl_BrOH_cation_2_vlapl() not generated due to NaN


def test_mgga_x_r2scanl_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
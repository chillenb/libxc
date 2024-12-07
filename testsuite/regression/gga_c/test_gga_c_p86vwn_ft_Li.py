
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_p86vwn_ft_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn_ft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.413370182496868e-02, -4.769677961175591e-02, 3.854556814124638e-03, -1.577813075237409e-02, -2.434095033394923e-03, -6.794961115159539e-03, -1.629259198827822e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_p86vwn_ft_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn_ft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.178331115980892e-01, -1.177156879803733e-01, -1.051625309749068e-01, -1.050756706350686e-01, -2.447187504085206e-02, -2.447628249166256e-02, -2.327912701054083e-02, -1.293579250969660e-01, -1.432330251938801e-02, -6.848326555774695e-02, -8.545032548981287e-03, -8.639541383792594e-03, -1.914214508247163e-04, -2.832624807333714e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_p86vwn_ft_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn_ft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.362202616896609e-05, 8.724405233793217e-05, 4.362202616896609e-05, 1.467225641609766e-04, 2.934451283219532e-04, 1.467225641609766e-04, 6.714371754824413e-03, 1.342874350964883e-02, 6.714371754824413e-03, 2.622121334579462e+00, 5.244242669158924e+00, 2.622121334579462e+00, 2.918484088762568e+01, 5.836968177525135e+01, 2.918484088762568e+01, -4.772717840252149e-03, -9.545435680504298e-03, -4.772717840252149e-03, -1.025999751632920e-29, -2.051999503265839e-29, -1.025999751632920e-29]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

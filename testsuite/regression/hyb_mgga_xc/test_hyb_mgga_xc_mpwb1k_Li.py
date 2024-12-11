
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data

# test_hyb_mgga_xc_mpwb1k_Li_2_zk() not generated due to NaN in reference data

# test_hyb_mgga_xc_mpwb1k_Li_2_vrho() not generated due to NaN in reference data

# test_hyb_mgga_xc_mpwb1k_Li_2_vsigma() not generated due to NaN in reference data


def test_hyb_mgga_xc_mpwb1k_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.873107274983154e-03, -2.871683944547766e-03, -5.332155022208856e-03, -5.324538201822170e-03, -5.428445283005290e-02, -5.432066785617735e-02, -1.392078133987678e-01, -1.176431803685671e-07, -5.378184861852190e-01, -3.855053355621430e-11, -5.743566782136254e-11, -1.256638198788993e-07, -5.361832595959912e-22, -6.047408678635331e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

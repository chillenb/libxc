
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data

# test_hyb_mgga_xc_mpw1kcis_Li_2_zk() not generated due to NaN in reference data

# test_hyb_mgga_xc_mpw1kcis_Li_2_vrho() not generated due to NaN in reference data

# test_hyb_mgga_xc_mpw1kcis_Li_2_vsigma() not generated due to NaN in reference data


def test_hyb_mgga_xc_mpw1kcis_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.517920489624387e-05, -3.517051707674886e-43, -2.978197219499551e-42, -2.968540450469716e-42, -1.405766216307531e-38, -1.481935180641130e-38, -1.022608013947658e-32, -2.443810795953947e-06, -1.385656088160181e-31, -1.572871022279231e-08, -1.151971136269993e-09, -2.527468106101991e-06, -3.203490006160380e-19, -7.541133187092537e-21])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

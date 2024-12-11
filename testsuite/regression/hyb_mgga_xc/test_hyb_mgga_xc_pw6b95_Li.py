
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data

# test_hyb_mgga_xc_pw6b95_Li_2_zk() not generated due to NaN in reference data

# test_hyb_mgga_xc_pw6b95_Li_2_vrho() not generated due to NaN in reference data

# test_hyb_mgga_xc_pw6b95_Li_2_vsigma() not generated due to NaN in reference data


def test_hyb_mgga_xc_pw6b95_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw6b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.873210903507369e-03, -2.871683944547766e-03, -5.332155022208856e-03, -5.324538201822170e-03, -5.428445283005290e-02, -5.432066785617735e-02, -1.392078133987678e-01, -1.262614296020221e-07, -5.378184861852190e-01, -4.137508707964219e-11, -6.164390033329540e-11, -1.348695534096974e-07, -5.754688095400736e-22, -6.047408679439781e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

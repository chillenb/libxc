
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data

# test_mgga_xc_vcml_rvv10_Li_2_zk() not generated due to NaN in reference data

# test_mgga_xc_vcml_rvv10_Li_2_vrho() not generated due to NaN in reference data

# test_mgga_xc_vcml_rvv10_Li_2_vsigma() not generated due to NaN in reference data


def test_mgga_xc_vcml_rvv10_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-4.894885142184958e-03, -4.931384796694587e-03, 3.042052432208864e-02, 3.039360480947706e-02, 2.104055089883573e-04, 2.262098499906099e-04, -1.080692202285795e-01, 1.529669417213426e-12, 8.666683933259188e-03, 2.397771671342468e-08, -3.384746550564411e-12, 4.925383620046366e-13, 4.399696475523100e-03, 2.671353916447233e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

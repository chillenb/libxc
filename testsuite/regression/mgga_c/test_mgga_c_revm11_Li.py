
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revm11_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.776336429992739e-02, -6.400692617450450e-02, -1.325763871602566e-01, 4.874428814346818e-03, -2.287311417222829e-02, -1.428935708334251e-02, -3.545721161640824e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revm11_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.911461789617149e-01, 2.910979425760042e-01, 6.407018652075561e-02, 6.417624010245307e-02, 3.586069018852116e-02, 3.574433190134129e-02, 5.651586419666323e-02, 8.435395469067397e-02, 6.738160557064279e-03, -1.151976741782209e-01, -1.795948550167984e-02, -1.816103412956522e-02, -4.171176054721493e-04, -6.120616475510716e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm11_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.935386183584595e-04, -1.787077236716919e-03, -8.935386183584595e-04, -5.445794910216952e-04, -1.089158982043390e-03, -5.445794910216952e-04, 4.757449581146663e-01, 9.514899162293325e-01, 4.757449581146663e-01, -2.617321453627587e+01, -5.234642907255174e+01, -2.617321453627587e+01, 3.559439414569906e+02, 7.118878829139811e+02, 3.559439414569906e+02, 6.384343147526381e-08, 1.276868664069541e-07, 6.384343147526381e-08, 6.724956979347340e-16, -1.189577865767141e-14, 6.724956979347340e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm11_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-3.702720789273725e-02, -3.702720789273725e-02, -1.971859772308669e-02, -1.971859772308669e-02, -3.444994452427833e-02, -3.444994452427826e-02, -1.380578923257935e+00, -1.380578923257634e+00, -2.854293728912626e-01, -2.854293726945134e-01, -3.304024921321247e-08, -3.304024921384644e-08, -8.731347932221727e-20, -8.729893919243842e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mvsbs_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.038110200317983e+00, -1.414830133240718e+00, -3.236729329362850e-01, -1.844517346928956e-01, -7.214900369517720e-02, -2.745569406055213e-03, -8.206942671049331e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mvsbs_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.826488252137899e+00, -2.828745230994066e+00, -2.130716062077705e+00, -2.131384366583255e+00, -4.466123925427458e-01, -4.436204164056440e-01, -2.506855552166435e-01, -5.134101828955673e-03, -8.577704597610053e-02, -3.655839079360388e-05, -5.109059826325262e-03, -5.440379054354206e-03, -1.764533267844838e-05, -1.202402515504535e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvsbs_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.018277059032018e-04, 0.000000000000000e+00, 4.993835542738805e-04, 3.436445174845318e-03, 0.000000000000000e+00, 3.402528118323623e-03, -3.257571057595413e-02, 0.000000000000000e+00, -3.950559730261193e-02, 6.711839046256907e+00, 0.000000000000000e+00, 7.225919899845833e+00, -7.730020521173792e+01, 0.000000000000000e+00, 4.075392300233926e+03, 1.407287967138650e+01, 0.000000000000000e+00, 6.557017331256243e+00, 1.280705792331222e+04, 0.000000000000000e+00, 1.277932288261763e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvsbs_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvsbs_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.597239762500162e-02, -2.591112454633377e-02, -5.956109651327395e-02, -5.903244606461773e-02, 1.508560292557289e-02, 1.728978712649505e-02, -2.595878572816048e-01, 1.344453334218985e-04, 2.085184107688510e-01, 2.417742178183649e-06, 3.787225133803467e-08, 1.387952066021707e-04, 4.000101523297816e-17, 8.117193937689759e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

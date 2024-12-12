
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m11_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.109736746033624e-01, -4.756274752955397e-01, -1.958580993503632e-01, -8.010472925909957e-03, -6.515230811210302e-03, -1.380071059267689e-04, -6.938098685072278e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m11_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([1.406902675696793e-02, 2.178103352032005e-02, -6.640511967787349e-01, -6.671295180554525e-01, -1.324909050741408e-01, -1.354102195722460e-01, 3.239421003852991e-02, -2.384509504983164e-04, -1.104035954183915e-02, -7.724328346992594e-09, -2.774617472714642e-04, -2.709624647853187e-04, -2.301900491268524e-09, -6.163415189447192e-25])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m11_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.593269422938509e-04, 0.000000000000000e+00, 5.606013180492451e-04, 7.049012172672238e-04, 0.000000000000000e+00, 7.045816781043757e-04, -4.437341117793343e-02, 0.000000000000000e+00, -4.402293021617884e-02, 1.107294142033037e+01, 0.000000000000000e+00, -8.581386926439291e-03, -6.102665636313743e+00, 0.000000000000000e+00, -5.571326238294919e-05, -9.676241392486639e-03, 0.000000000000000e+00, -8.868283343561394e-03, -1.809471992980608e-05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m11_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.667371316342367e-01, -1.679911191102004e-01, -1.795906338346387e-02, -1.763835964759924e-02, -1.329901951160348e-02, -1.291047751719123e-02, -1.948830829130747e+00, -4.674625814326391e-08, 1.183171434773588e-02, -9.651019187353149e-15, -2.535363726209763e-11, -5.496870233118154e-08, -5.929139473467922e-26, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

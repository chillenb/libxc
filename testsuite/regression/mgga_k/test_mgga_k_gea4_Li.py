
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_gea4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.604232847009168e+02, 4.231389342270655e+00, 3.507381192906101e+00, 3.878296221686046e-02, 7.095653121909649e-02, 1.413871498245486e+05, 7.510532178286925e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_gea4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.362275075541214e+02, -1.358896103988943e+02, 1.194283281060677e+01, 1.196429229056922e+01, 3.870756621157505e-01, 3.745373489368515e-01, 2.086017194617672e-01, 6.778290298939235e+01, 2.396838057933952e-02, -1.668152268386230e+04, -4.529778085484995e+05, 6.428193587704897e+01, -1.701476465563223e+15, -1.107990541802466e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_gea4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.220739278061128e-02, 0.000000000000000e+00, 1.215643968536223e-02, 8.035054102335463e-03, 0.000000000000000e+00, 8.013252362457831e-03, 4.102754284450041e-01, 0.000000000000000e+00, 4.280648447381405e-01, 3.709360567367293e+00, 0.000000000000000e+00, -4.575350847237157e+06, 5.550551316784406e+01, 0.000000000000000e+00, 2.589214828659871e+13, -2.874663356315814e+08, 0.000000000000000e+00, -3.410122758908141e+06, -1.056123043706440e+20, 0.000000000000000e+00, 1.592783284771492e+15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_gea4_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([5.066728318415874e-02, 5.087544309861060e-02, 1.604931436040591e-01, 1.604980812858715e-01, 1.713544957915547e-01, 1.704393833726190e-01, 1.597567517439439e-01, 1.449037670729314e+01, 1.628008973084329e-01, -1.827175517441046e+03, 9.500833298804457e+02, 1.243903145704640e+01, 2.849563421861375e+09, -1.319436924733016e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

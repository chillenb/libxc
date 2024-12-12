
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_hle17_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.485813004204159e+00, -1.729070259301069e+00, -5.019768590095663e-01, -2.222912666123025e-01, -9.581251290439707e-02, -2.568060314802715e-02, -4.798233723307655e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_hle17_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.061623095305248e+00, -3.064322341764219e+00, -2.177127054165768e+00, -2.179297863302149e+00, -4.477945790006068e-01, -4.487033706478987e-01, -2.818449963629852e-01, -2.307117554363744e-01, -9.452435158209212e-02, -5.871196294271138e-03, -3.432150623907639e-02, -3.407489519730397e-02, -6.926944107902533e-04, -4.924427519638483e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_hle17_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.196192236653160e-03, 9.667814540615128e-05, -1.195104209678319e-03, -1.873518837458683e-03, 1.594185460440273e-04, -1.870095927514583e-03, -1.036798086816795e-01, -1.168554007977317e-02, -1.028378742255846e-01, -1.860052615652254e+01, 4.435900337745596e+01, 2.274477665269284e+02, -7.195897491879636e+01, 1.572290657416166e+01, 3.126885594107182e+04, -3.530220358705184e-01, -1.039570271036991e-04, -3.296261231032200e-01, -1.616527707692322e+00, 1.606947892060015e-06, -2.313892433039144e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_hle17_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([6.055120031546056e-02, 6.071151676084023e-02, 3.205510197396630e-02, 3.215526556674562e-02, 9.258978217400113e-04, 8.208634251389729e-04, 6.967974286405925e-01, -5.685772307638325e-01, -2.856598836117063e-03, -1.881312249715314e-02, 1.308317788976601e-14, 9.622610712040926e-11, -4.535211274036836e-32, 7.551374569862903e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

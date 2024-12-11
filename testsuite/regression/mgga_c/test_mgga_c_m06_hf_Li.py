
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m06_hf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.883580834794238e-02, -7.566274191679782e-02, -9.775319363174437e-02, 1.611235804420147e-02, -5.753949090564558e-02, -4.234804117235515e-02, -7.625564167078678e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m06_hf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-7.807162696120901e-02, -7.711299606299396e-02, -6.806296060242845e-02, -6.747450831651358e-02, -4.895135346639140e-02, -4.095180693334458e-02, 3.141114891833419e-02, -1.540275185089506e+00, 1.649245593956741e-02, -9.271260638155655e-01, -1.061934797749586e-01, -1.071087403552734e-01, -2.320013253364070e-03, -2.315735039705950e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_hf_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.247110459003147e-04, 0.000000000000000e+00, 4.301511224871773e-04, 3.368306680797363e-03, 0.000000000000000e+00, 3.352344796854462e-03, 3.741736890275932e+00, 0.000000000000000e+00, 3.738114599254698e+00, 1.736140566205989e+01, 0.000000000000000e+00, 1.035246798324555e+03, 4.982375288906985e+03, 0.000000000000000e+00, 5.233437044199790e+06, 2.397240332746881e+01, 0.000000000000000e+00, 1.143247897413194e+03, 8.151632992829026e+01, 0.000000000000000e+00, 7.317490009199975e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_hf_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.236301618555245e-03, -1.277421064769818e-03, -2.277532513089777e-03, -2.305741833551121e-03, -1.180593893408672e-02, -1.294060892185174e-02, -3.459069156780987e-01, -2.028232555681357e-02, -7.141908413853956e-01, -2.159004716975858e-03, -3.561879186043320e-04, -1.661097676285288e-02, -9.897391144751950e-09, 8.563314822341994e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

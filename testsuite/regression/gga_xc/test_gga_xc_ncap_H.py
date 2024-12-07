
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_ncap_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_ncap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.542126295379299e-01, -5.967742508126334e-01, -3.680109285650304e-01, -1.488290607047975e-01, -3.204071058150674e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_ncap_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_ncap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.652048224185712e-01, -2.218122288464846e-01, -7.617178441131928e-01, -2.122232923295846e-01, -4.360096979793419e-01, -1.661943436249679e-01, -8.275035052663306e-02, -4.896711856436604e-02, 1.529146166964063e-01, -7.162840556528678e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_ncap_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_ncap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.905191790885356e-03, 3.017758821746206e-02, 1.508879410873103e-02, -1.307138961597754e-02, 2.393721012269707e-02, 1.196860506134854e-02, -1.115333768149627e-01, 1.436775279986283e-01, 7.183876399931417e-02, -1.281289850250356e+01, 2.122729652063161e+00, 1.061364826031581e+00, -4.634939967240014e+05, -2.131036235152263e+00, -1.065518117576132e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

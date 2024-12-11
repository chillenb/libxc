
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_wl_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.553327595043493e-07, -2.831109513527548e-07, -3.436172391290043e-07, 1.247389525332349e-06, 3.412826154202750e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_wl_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.137863372166462e-07, -9.588629844595918e+04, -4.154460882283771e-07, -3.906374715896061e+04, -7.817426189326845e-07, -8.943004559480565e+03, -2.177010133724029e-06, 5.509908559834145e+02, 7.934542980943139e-05, 1.466924616068837e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_wl_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([9.492120040759069e-07, 6.627829941448628e-07, 1.050450607456753e+24, 1.218470295389892e-07, 1.228185274721089e-07, 3.240738688364738e+23, 1.293217306676739e-06, 1.685706978362247e-06, 5.830424699848851e+22, 3.254109365540269e-04, 8.887344403594110e-04, -1.505834325782206e+21, 9.348684827146853e+01, 7.447321709199880e+02, -2.110335375567033e+18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

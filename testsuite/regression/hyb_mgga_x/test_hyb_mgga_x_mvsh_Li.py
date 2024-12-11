
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_mvsh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.479012125823747e+00, -1.000050472631082e+00, -1.930904571413778e-01, -1.352071081062104e-01, -4.233895906539755e-02, -1.519162722165131e-03, -2.864743025897456e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_mvsh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.039987661392048e+00, -2.041891070216886e+00, -1.427707516510141e+00, -1.428516541131729e+00, -2.632741230709695e-01, -2.634142534187128e-01, -1.846825527518953e-01, 1.509171296198760e+00, -5.903251137783370e-02, 4.240993180701829e+00, 2.712200330255866e+01, 1.506043642310819e+00, 3.915734897338394e+04, -1.429794929702161e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mvsh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.891988634464441e-04, 0.000000000000000e+00, -1.885734958147175e-04, -7.994702634009504e-04, 0.000000000000000e+00, -7.937420249232806e-04, -4.393503600424177e-03, 0.000000000000000e+00, -4.615902829908194e-03, -3.221894565450724e+00, 0.000000000000000e+00, -3.874636942400716e+04, -9.034971179081420e+00, 0.000000000000000e+00, -8.600911078580427e+09, -1.231691011151460e+04, 0.000000000000000e+00, -3.311701939394402e+04, -1.800359197892764e+09, 0.000000000000000e+00, 7.275914500810837e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mvsh_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [9.817400459275441e-03, 9.811350640858019e-03, 1.388368894239146e-02, 1.381943868091997e-02, 1.057807212391008e-03, 1.109936153366871e-03, 1.236796305496829e-01, 4.950270338032584e-01, 2.160691786238279e-02, 3.504318829085345e+00, 1.829219982167708e-01, 4.813487182055661e-01, 2.185925102028765e-01, 1.567428299559763e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

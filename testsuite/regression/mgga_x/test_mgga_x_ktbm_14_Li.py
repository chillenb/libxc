
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_14_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_14", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.157202153461203e+00, -1.526596509539933e+00, -2.655837228563185e-01, -1.918784063228895e-01, -6.476368193891953e-02, -9.181934555990569e-03, -1.719166558965723e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_14_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_14", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.582803363969390e+00, -2.585277198776295e+00, -1.753518468717749e+00, -1.754677302669619e+00, -3.641150552525310e-01, -3.668080318760168e-01, -2.391708380101685e-01, -1.123376791857520e-02, -9.166164344864354e-02, -3.561813296520849e-04, -1.233377557933443e-02, -1.172736102841423e-02, -2.485557134749409e-04, -1.691312336594183e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_14_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_14", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.470359817718146e-04, 0.000000000000000e+00, -8.441600178405392e-04, -3.321946819172221e-03, 0.000000000000000e+00, -3.313223406484038e-03, -5.662392085882363e-02, 0.000000000000000e+00, -5.996028644880679e-02, -1.291217454828985e+01, 0.000000000000000e+00, -1.073554668518660e+01, -9.262430102665729e+01, 0.000000000000000e+00, -2.682379813272728e+04, 2.699777714438956e-01, 0.000000000000000e+00, -9.600274992501229e+00, 5.740364732381951e-01, 0.000000000000000e+00, -1.214349133986645e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_14_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_14", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.873699452428417e-02, 2.869543738482772e-02, 5.111403773682360e-02, 5.106455586233149e-02, 2.326410909729949e-02, 2.518780775818490e-02, 2.784689130190017e-01, 1.376944107957849e-04, 3.988749252990531e-01, 1.092908369282483e-05, -8.206496987422709e-08, 1.401167874335459e-04, -5.518882032992065e-16, 5.297391140400372e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

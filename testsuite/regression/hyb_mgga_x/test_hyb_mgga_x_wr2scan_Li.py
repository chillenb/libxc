
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_wr2scan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.845713302526541e+00, -1.224612156562046e+00, -1.744595039517408e-01, -5.690178863655695e-02, -5.409190971903487e-03, -1.662352569897658e-05, -1.658333379139539e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_wr2scan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.480127296105701e+00, -2.482601411512089e+00, -1.643356722446383e+00, -1.645010536240951e+00, -1.747101265642910e-01, -2.022205021889370e-01, -9.725793484722342e-02, 7.811112269811121e-04, -9.383374853988123e-03, 8.658473002318141e-09, -3.680523319466062e-05, 8.880089012308674e-04, -7.229521649718660e-11, -1.033490513518219e-26]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_wr2scan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.992678013223592e-04, 0.000000000000000e+00, -1.985721125597653e-04, -7.435814908647406e-04, 0.000000000000000e+00, -7.404147561228383e-04, -1.387262915920400e-01, 0.000000000000000e+00, -1.019514216641582e-01, -1.096394246545630e+00, 0.000000000000000e+00, -2.094112687216602e+01, -7.122703329175526e+00, 0.000000000000000e+00, -1.826641629996410e+01, 5.765488541128182e-02, 0.000000000000000e+00, -2.039654610279709e+01, 3.761566188301135e-02, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_wr2scan_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_wr2scan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.068042473165790e-02, 1.067211011426342e-02, 1.360022914675739e-02, 1.357921904969342e-02, 3.490307563501952e-02, 2.609439854999005e-02, 4.277396035685913e-02, 2.683684734677235e-04, 1.833226675391673e-02, 7.470339159499810e-09, 3.068513057224427e-13, 2.973696326329880e-04, 1.414390596414763e-25, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

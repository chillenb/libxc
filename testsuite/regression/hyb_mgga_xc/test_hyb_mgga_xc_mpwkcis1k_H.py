
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_mpwkcis1k_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.669382962764645e-01, -3.424581622229194e-01, -2.133595831817067e-01, -8.422210101012904e-02, -2.260115400404197e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_mpwkcis1k_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.205953279427363e-01, -2.236903732310476e-01, -4.452415654894433e-01, -1.870772614507487e-01, -2.517377360884935e-01, -1.295847842555173e-01, -7.158818294924096e-02, -2.851872973089982e-02, -8.374213313682015e-04, -9.920875263050759e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwkcis1k_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.296957245973598e+00, 4.142665982842431e-02, 8.608116618147828e+13, 1.110155752817838e-02, 1.331120876263109e-02, 8.596311204654973e+13, -2.527626213902716e-02, 6.124766703732045e-02, 8.612673461976172e+13, -4.212071871500357e+00, 8.818506781180349e-01, 8.613032435712288e+13, 4.346713531150850e+02, 2.009733486597134e+01, 8.613054221073778e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwkcis1k_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-3.114511462711588e+00, -3.995554316157371e-04, -4.506598657184718e-02, -3.990074693494971e-04, -2.544624488301368e-02, -3.997669429258967e-04, -3.879145572284060e-03, -3.997836051175632e-04, -6.913882799401646e-06, -3.997846163109039e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

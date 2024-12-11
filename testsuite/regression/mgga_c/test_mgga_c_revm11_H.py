
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revm11_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.274212845031208e-02, -1.802649683619834e-02, -2.516526507389561e-02, -6.764957488667146e-02, -1.268703772433631e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revm11_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.817545764682320e-02, -5.341671950435136e+02, 4.809014223954716e-02, -7.650344089162679e-02, -2.662497193888547e-02, -1.928892930598666e-01, -3.142002061294074e-02, -4.126957961124404e-01, -1.616037788939748e-02, -5.713298506936722e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm11_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.718298996800017e-01, -9.436597993600034e-01, -4.718298996800017e-01, -4.922876153334135e-02, -9.845752306668271e-02, -4.922876153334135e-02, 2.530754054907384e-01, 5.061508109814767e-01, 2.530754054907384e-01, 1.918576793401193e+02, 3.837153586802386e+02, 1.918576793401193e+02, 5.322340243164492e+07, 1.064468048632898e+08, 5.322340243164492e+07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm11_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.157217494495129e+00, 1.156634342909633e+00, -8.736674869287006e-02, -8.678131786368122e-02, -2.493685474436598e-03, -2.488674305874447e-03, -5.745362353410282e-02, -5.745327938839393e-02, -1.493759577345695e-05, -1.493759585916183e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

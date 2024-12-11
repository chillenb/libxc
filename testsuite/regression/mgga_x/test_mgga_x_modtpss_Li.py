
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_modtpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_modtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.750859403738739e+00, -1.209543471002415e+00, -2.977463914237423e-01, -1.585801933775156e-01, -6.345454625449135e-02, -2.055156836598811e-02, -3.505641557990653e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_modtpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_modtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.349540592563677e+00, -2.351673745982466e+00, -1.628193158117273e+00, -1.629613442017412e+00, -3.936157016237703e-01, -3.932657660380642e-01, -2.124450880323105e-01, -2.612127983609464e-02, -8.274698585740911e-02, -8.296437722074185e-04, -2.750719138509135e-02, -2.726608788142596e-02, -5.541564195127542e-04, -2.260030367871270e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_modtpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_modtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.448252753792544e-04, 0.000000000000000e+00, -3.422734052682319e-04, -9.643639593758076e-04, 0.000000000000000e+00, -9.623626053467698e-04, -4.611793034015238e-02, 0.000000000000000e+00, -4.680196737521956e-02, -7.486806281462538e+00, 0.000000000000000e+00, -2.422948329640078e-01, -3.462072782396304e+01, 0.000000000000000e+00, -1.547458302426394e+00, -1.030810347684407e-04, 0.000000000000000e+00, -2.299511353474991e-01, -7.063142234951648e-11, 0.000000000000000e+00, -2.295781757302868e+11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_modtpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_modtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.323739117125071e-03, 2.304817011437339e-03, 2.364145176797173e-03, 2.368349765299592e-03, -6.355538362972272e-04, -6.726087519172267e-04, 2.683296950295916e-02, 5.718191775165312e-11, -1.556610338874278e-02, 2.959889126178765e-17, 6.448471468791149e-16, 6.554609955264217e-11, 1.213527453049680e-33, -5.495081988234059e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

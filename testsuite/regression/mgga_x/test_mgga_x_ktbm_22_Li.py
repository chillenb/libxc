
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_22_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.189472962294593e+00, -1.537844645798921e+00, -2.797468241375060e-01, -1.954995234209477e-01, -6.659632472771976e-02, -1.003728089831184e-02, -1.887716686838480e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_22_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.650444241046269e+00, -2.652998532202725e+00, -1.788387142796017e+00, -1.789646234786838e+00, -3.726803071971677e-01, -3.748482668388607e-01, -2.455990476738958e-01, -1.226468745946187e-02, -9.030845212349187e-02, -3.889113161742856e-04, -1.358135255133568e-02, -1.280348924668040e-02, -2.737280705008236e-04, -1.846729734253496e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_22_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.826758855755094e-04, 0.000000000000000e+00, -8.797015608494955e-04, -3.435159481041507e-03, 0.000000000000000e+00, -3.426330301118172e-03, -5.790819157501318e-02, 0.000000000000000e+00, -6.155306680403051e-02, -1.350130078813303e+01, 0.000000000000000e+00, -9.752721938390247e+00, -9.540853338214085e+01, 0.000000000000000e+00, -2.434501569074071e+04, 3.934743684843655e-01, 0.000000000000000e+00, -8.721891201120027e+00, 8.314256302016398e-01, 0.000000000000000e+00, -1.102129112510229e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_22_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_22_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.520171573019153e-02, 3.515990453738502e-02, 5.716663431522365e-02, 5.713125875376646e-02, 2.185681776956431e-02, 2.379440375230195e-02, 3.583981775046480e-01, 1.250430128667067e-04, 3.777386233312455e-01, 9.919119811369006e-06, -1.196476762474303e-07, 1.272473689420676e-04, -7.993466455720744e-16, 4.807849396652597e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

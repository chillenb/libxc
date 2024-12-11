
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_tuned_cam_b3lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_tuned_cam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.640257275713540e+00, -1.159739273640381e+00, -2.715869815831924e-01, -8.604645364803416e-02, -1.631159858801367e-02, -3.092201134503780e-03, -5.528727845253322e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_tuned_cam_b3lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_tuned_cam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.063867980926795e+00, -2.065641746754920e+00, -1.407160331159510e+00, -1.408239143300124e+00, -3.749123427113146e-01, -3.752058327623680e-01, -1.276773597853821e-01, -9.130932774661164e-02, -2.688825706899034e-02, -3.867558986067693e-02, -4.038487112334032e-03, -4.125082225778198e-03, -5.321748379629024e-05, -1.296433707820868e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_tuned_cam_b3lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_tuned_cam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.283881825158913e-04, 4.230480491699886e-06, -2.277321820024006e-04, -8.155466407609908e-04, 2.954022849291356e-05, -8.135049299486096e-04, -3.047423762051717e-02, 3.866747919504811e-02, -3.029786248763092e-02, -1.579509176507002e+00, 3.722869163256963e+00, 2.792130281159698e+00, -3.483591204073989e+00, 1.909121184807025e+01, 1.431841019743528e+01, 3.172144660678506e-02, 6.428238830639903e-02, 3.198019914356233e-02, -6.591185228363022e-08, 0.000000000000000e+00, -3.188571285855826e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

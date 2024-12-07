
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b5050lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b5050lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.586479950477204e-01, -6.946718723561545e-01, -2.026683297237869e-01, -8.344147313102325e-02, -4.119258852679527e-02, -5.983238088732315e-02, -2.259018579251501e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b5050lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b5050lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.198251586255710e+00, -1.199141955797965e+00, -8.389680232292855e-01, -8.394859423698245e-01, -2.688951095914172e-01, -2.691811967728022e-01, -1.068736832293022e-01, -1.073093802758975e-01, -3.993295041364898e-02, -4.195656143478640e-02, -2.041990353842599e-02, -2.057670953849689e-02, -3.211509573509597e-03, -2.857414160269556e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b5050lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b5050lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.097196338158466e-04, 4.230480491699886e-06, -1.094395401303399e-04, -3.999833729254302e-04, 2.954022849291356e-05, -3.991581114707473e-04, -2.013204670226610e-02, 3.866747919504811e-02, -1.998957337782088e-02, -1.852289163404703e+00, 3.722869163256963e+00, -5.598343919408019e+02, -3.216997392870944e+01, 1.909121184807025e+01, -2.037125955113112e+07, -4.892109976315116e+02, 6.428238830639903e-02, -4.899915565645902e+02, -6.048009393596680e+07, 0.000000000000000e+00, -1.801636714007792e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

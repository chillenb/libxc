
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_b00_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_b00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.710618602228752e+00, -1.486475266894497e+00, -4.233427929364385e-01, -1.697094194520613e-01, -6.533357380016781e-02, -1.681406059767928e-01, -4.498275941926377e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_b00_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_b00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.993392872081094e+00, -1.993439953769746e+00, -2.558484519092294e+00, -2.558519454519177e+00, -2.317531020750887e-01, -2.279850164265340e-01, -2.163008687728918e-01, -7.646409264309025e-02, -4.413804224141142e-02, -6.783089043531676e-03, -7.636357511474295e-02, -7.604667840655865e-02, -3.583135351455157e-03, -1.532172015654800e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_b00_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_b00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.560589286414931e-07, 0.000000000000000e+00, -9.557853288514195e-07, -1.520879769415357e-03, 0.000000000000000e+00, -1.517405284876737e-03, -8.838889968699021e-02, 0.000000000000000e+00, -8.798663558810446e-02, -2.975011208378036e+00, 0.000000000000000e+00, -9.485278718149297e+02, -6.013434752025894e+01, 0.000000000000000e+00, -4.620739495409255e+07, -8.347087963219356e+02, 0.000000000000000e+00, -8.431967329247772e+02, -1.439269995871519e+08, 0.000000000000000e+00, -4.409896219308593e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_b00_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_b00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.240221941928710e-05, -1.243221503402946e-05, -6.602941536409933e-03, -6.604692907145992e-03, -5.320265105460290e-03, -5.289298293910118e-03, -2.855061514729357e-02, -3.028578220305791e-03, -3.595246409270557e-02, -4.706632279095789e-03, -3.099101231658621e-03, -3.062803142500111e-03, -4.368762100705400e-03, -4.809344240464708e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_b00_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_b00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-4.549339662262764e-02, -4.573625590721943e-02, 8.806145235591523e-02, 8.770047117170826e-02, -1.269757225978529e-03, -2.364679938474205e-03, -1.079282049066695e+00, 1.211424548780962e-02, -3.502345445944742e-03, 1.882652911606666e-02, 1.239640492591066e-02, 1.225113811767389e-02, 1.747504840282160e-02, 1.923737696184455e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

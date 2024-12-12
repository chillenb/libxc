
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_cc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.348327888741848e-02, -8.371468637109090e-02, -4.959804181920233e-02, -1.553680552883682e-03, -2.766210798901375e-08, -6.778618832875420e-03, -1.681734331940153e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_cc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.028956119607139e-01, -1.022286856239165e-01, -9.275917972041750e-02, -9.219341657441744e-02, -5.658301259076715e-02, -5.675134409699735e-02, -1.831587533639474e-02, -9.336338016192129e-02, -1.095911155168896e-02, -5.479561785049036e-02, -8.515775240750634e-03, -8.623355041107481e-03, -1.978382598429454e-04, -2.903034120678790e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.891865759421425e-10, 7.783731518842850e-10, 3.891865759421425e-10, 9.555965868571372e-10, 1.911193173714274e-09, 9.555965868571372e-10, 1.286395797951029e-08, 2.572791595902057e-08, 1.286395797951029e-08, 2.128378562884594e+01, 4.256757125769189e+01, 2.128378562884594e+01, 6.394933884360409e+01, 1.278986776872082e+02, 6.394933884360409e+01, 3.615368343344791e-04, 7.230736686689582e-04, 3.615368343344791e-04, 1.717940799906916e+00, 3.435881599813832e+00, 1.717940799906916e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cc_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-4.043339845526006e-08, -4.043339845526004e-08, -3.321048083912594e-08, -3.321048083912593e-08, -6.103314375637578e-09, -6.103314375637576e-09, -7.480830913226326e-01, -7.480830913224695e-01, -1.529330612905926e-01, -1.529330611851744e-01, -4.282337510868058e-10, -4.282337510868059e-10, -2.914907304162560e-15, -2.914907304162561e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

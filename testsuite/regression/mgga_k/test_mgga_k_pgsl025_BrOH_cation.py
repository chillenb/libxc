
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data

# test_mgga_k_pgsl025_BrOH_cation_2_zk() not generated due to NaN

# test_mgga_k_pgsl025_BrOH_cation_2_vrho() not generated due to NaN


def test_mgga_k_pgsl025_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pgsl025", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.314865541902636e-06, 0.000000000000000e+00, 5.314877473105034e-06, 5.314706214880409e-06, 0.000000000000000e+00, 5.314761789545037e-06, 5.314330532095072e-06, 0.000000000000000e+00, 5.314185942894398e-06, 5.316126501614160e-06, 0.000000000000000e+00, 5.316476178858566e-06, 5.314763748014359e-06, 0.000000000000000e+00, 5.315659216268295e-06, 5.314763748014359e-06, 0.000000000000000e+00, 5.315659216268295e-06, 1.489471120426001e-03, 0.000000000000000e+00, 1.488416474945260e-03, 1.489317469627951e-03, 0.000000000000000e+00, 1.488200418330173e-03, 1.484959295434881e-03, 0.000000000000000e+00, 1.483790099061985e-03, 1.491789442733719e-03, 0.000000000000000e+00, 1.490500735395748e-03, 1.490658468047506e-03, 0.000000000000000e+00, 1.486574387146463e-03, 1.490658468047506e-03, 0.000000000000000e+00, 1.486574387146463e-03, 2.803582423563805e-01, 0.000000000000000e+00, 2.696878316222881e-01, 2.841909951814345e-01, 0.000000000000000e+00, 2.710829265545272e-01, 3.359074929160683e-01, 0.000000000000000e+00, 3.544990299972074e-01, 3.377053440019367e-01, 0.000000000000000e+00, 3.318142528529749e-01, 2.525127927908552e-01, 0.000000000000000e+00, 4.308599909078028e-01, 2.525127927908552e-01, 0.000000000000000e+00, 4.308599909078028e-01, 2.539588637151878e+01, 0.000000000000000e+00, 2.249148097761870e+01, 2.438720960290815e+01, 0.000000000000000e+00, 2.120571420906932e+01, 1.823630593831302e-01, 0.000000000000000e+00, 1.472504405000895e-01, 5.658987408307299e+01, 0.000000000000000e+00, 5.430887039765022e+01, 2.227048342638913e+01, 0.000000000000000e+00, 1.463508588786506e+02, 2.227048342638912e+01, 0.000000000000000e+00, 1.463508588786506e+02, 6.324503014234192e+05, 0.000000000000000e+00, 5.271019733892687e+05, 5.496562727421005e+05, 0.000000000000000e+00, 4.471116735888456e+05, 3.197159921025621e+03, 0.000000000000000e+00, 2.714904401504723e+03, 2.911560192811907e+06, 0.000000000000000e+00, 3.062094217456302e+06, 8.908246190359766e+05, 0.000000000000000e+00, 4.834051151218384e+06, 8.908246190359766e+05, 0.000000000000000e+00, 4.834051151218380e+06, 3.043509033941792e-04, 0.000000000000000e+00, 3.046280200123773e-04, 3.027461613529956e-04, 0.000000000000000e+00, 3.030759511630018e-04, 3.042671387391807e-04, 0.000000000000000e+00, 3.045785483817162e-04, 3.028900839224659e-04, 0.000000000000000e+00, 3.031684933842764e-04, 3.035244341568674e-04, 0.000000000000000e+00, 3.038475014282835e-04, 3.035244341568674e-04, 0.000000000000000e+00, 3.038475014282835e-04, 1.309829910006728e-02, 0.000000000000000e+00, 1.310061309573974e-02, 1.269365136890142e-02, 0.000000000000000e+00, 1.270687718342416e-02, 1.366490888883155e-02, 0.000000000000000e+00, 1.350170801337262e-02, 1.330181964513014e-02, 0.000000000000000e+00, 1.313982180211711e-02, 1.246067089348064e-02, 0.000000000000000e+00, 1.283460113517374e-02, 1.246067089348064e-02, 0.000000000000000e+00, 1.283460113517374e-02, 2.831723604531490e-01, 0.000000000000000e+00, 2.849563948485893e-01, 1.084203127721542e-01, 0.000000000000000e+00, 1.067963883068254e-01, 4.206416326329914e-01, 0.000000000000000e+00, 3.581506379681265e-01, 1.889160704232808e-01, 0.000000000000000e+00, 1.738151322628165e-01, 2.321991617595898e-01, 0.000000000000000e+00, 2.780905632843436e-01, 2.321991617595900e-01, 0.000000000000000e+00, 2.780905632843435e-01, 1.680322501718963e+02, 0.000000000000000e+00, 1.633153647255778e+02, 2.429561890532178e+01, 0.000000000000000e+00, 2.393349037244075e+01, 2.269837421707443e+02, 0.000000000000000e+00, 1.940019561775987e+02, 3.832835380601127e-03, 0.000000000000000e+00, 3.835189495597517e-03, 1.121634140249295e+02, 0.000000000000000e+00, 9.435052051172438e+01, 1.121634140249295e+02, 0.000000000000000e+00, 9.435052051172438e+01, 6.896246147170695e+06, 0.000000000000000e+00, 6.144157416218540e+06, 3.268601510341824e+06, 0.000000000000000e+00, 3.124571891844912e+06, 7.817219853849012e+06, 0.000000000000000e+00, 6.581441351138420e+06, 6.939684423521389e+02, 0.000000000000000e+00, 6.813232572463984e+02, 7.337213909483019e+06, 0.000000000000000e+00, 2.832798288875731e+06, 7.337213909483018e+06, 0.000000000000000e+00, 2.832798288875731e+06, 9.328263577353801e-02, 0.000000000000000e+00, 9.151199515894801e-02, 1.366313724726786e-01, 0.000000000000000e+00, 1.338280719853241e-01, 1.215586379658634e-01, 0.000000000000000e+00, 1.190170110352297e-01, 1.088469442674388e-01, 0.000000000000000e+00, 1.067928505909709e-01, 1.152112282714474e-01, 0.000000000000000e+00, 1.129184372546416e-01, 1.152112282714474e-01, 0.000000000000000e+00, 1.129184372546416e-01, 8.718543255572984e-02, 0.000000000000000e+00, 8.621676390310644e-02, 6.915476899663140e-01, 0.000000000000000e+00, 6.780114904512690e-01, 4.788145226916256e-01, 0.000000000000000e+00, 4.675501249611038e-01, 2.956729842383216e-01, 0.000000000000000e+00, 2.910538559009553e-01, 3.827534299798173e-01, 0.000000000000000e+00, 3.767230302188577e-01, 3.827534299798173e-01, 0.000000000000000e+00, 3.767230302188577e-01, 9.886024617149800e-02, 0.000000000000000e+00, 9.642252460770953e-02, 1.194187055820974e+01, 0.000000000000000e+00, 1.172855195423387e+01, 6.650309886995337e+00, 0.000000000000000e+00, 6.417840149459259e+00, 2.123284675929348e+00, 0.000000000000000e+00, 2.077433025456897e+00, 3.873368798388030e+00, 0.000000000000000e+00, 3.867080634745529e+00, 3.873368798388034e+00, 0.000000000000000e+00, 3.867080634745530e+00, 9.045180541077454e-01, 0.000000000000000e+00, 8.749077040216444e-01, 3.383498365853439e+03, 0.000000000000000e+00, 3.314985802507890e+03, 1.390172763857115e+03, 0.000000000000000e+00, 1.259415447184266e+03, 2.044525856382108e+00, 0.000000000000000e+00, 1.910524559464577e+00, 3.625707363724385e+02, 0.000000000000000e+00, 3.079851680515123e+02, 3.625707363724384e+02, 0.000000000000000e+00, 3.079851680515124e+02, 2.160802635499018e+05, 0.000000000000000e+00, 1.941851121390783e+05, 1.678524443444068e+08, 0.000000000000000e+00, 1.667067706068541e+08, 1.992427887587956e+07, 0.000000000000000e+00, 1.656041588063657e+07, 4.122466915137452e+02, 0.000000000000000e+00, 3.926655637176709e+02, 8.633543549065117e+06, 0.000000000000000e+00, 3.678861720624852e+06, 8.633543549065135e+06, 0.000000000000000e+00, 3.678861720624863e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

# test_mgga_k_pgsl025_BrOH_cation_2_vlapl() not generated due to NaN


def test_mgga_k_pgsl025_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pgsl025", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
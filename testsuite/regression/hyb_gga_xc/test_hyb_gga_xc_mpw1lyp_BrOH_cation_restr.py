
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mpw1lyp_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1lyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.586757997449956e+01, -1.586760009259392e+01, -1.586774157300705e+01, -1.586744118237270e+01, -1.586759118081538e+01, -1.586759118081538e+01, -2.678833585306073e+00, -2.678805711962553e+00, -2.678225709379991e+00, -2.679733069484272e+00, -2.678863503757921e+00, -2.678863503757921e+00, -5.623352382521896e-01, -5.622159904095494e-01, -5.602282506109245e-01, -5.641105273187554e-01, -5.627906784082840e-01, -5.627906784082840e-01, -1.671981190588815e-01, -1.689051625113823e-01, -6.468331900845127e-01, -1.338027510562587e-01, -1.470658970422182e-01, -1.470658970422183e-01, -1.685752554604351e-03, -1.854882786507341e-03, -4.123637330473675e-02, -7.436688800931848e-04, -1.044159926394472e-03, -1.044159926394471e-03, -3.858122150543161e+00, -3.857610508827819e+00, -3.858106955197522e+00, -3.857655159579101e+00, -3.857858272658965e+00, -3.857858272658965e+00, -1.639627602834016e+00, -1.647455334456091e+00, -1.639812510498200e+00, -1.646731595920420e+00, -1.643853519023669e+00, -1.643853519023669e+00, -4.764028079506566e-01, -4.962559058371402e-01, -4.451009804355005e-01, -4.448618065133738e-01, -4.813856981731955e-01, -4.813856981731955e-01, -9.712532725649581e-02, -1.712330924307155e-01, -9.139735360093343e-02, -1.421321122689574e+00, -1.127955959999362e-01, -1.127955959999362e-01, -5.455466576460926e-04, -7.314224964416559e-04, -6.021935675691828e-04, -6.894971328074259e-02, -7.228686879346000e-04, -7.228686879346001e-04, -4.556524808111065e-01, -4.580847886785414e-01, -4.573053754084369e-01, -4.566068511902482e-01, -4.569625508699067e-01, -4.569625508699067e-01, -4.423811892845323e-01, -4.168606903516299e-01, -4.253459569647789e-01, -4.327036608954814e-01, -4.289396953127507e-01, -4.289396953127507e-01, -5.197999415216720e-01, -2.107829358134328e-01, -2.450043990553927e-01, -3.010656257898766e-01, -2.716163296580594e-01, -2.716163296580593e-01, -3.856453416593339e-01, -3.791468404879027e-02, -5.462842345839613e-02, -2.851028285855005e-01, -8.382982257445101e-02, -8.382982257445103e-02, -2.968751508102580e-03, -1.678265208089677e-04, -3.762988205652806e-04, -8.004694742783457e-02, -6.639387832852732e-04, -6.639387832852728e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mpw1lyp_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1lyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.898324405538353e+01, -1.898331459683765e+01, -1.898361593289696e+01, -1.898256668017086e+01, -1.898311820758048e+01, -1.898311820758048e+01, -3.116570520255673e+00, -3.116597763159998e+00, -3.117302881144265e+00, -3.116656612373299e+00, -3.116676668474192e+00, -3.116676668474192e+00, -6.395739791181020e-01, -6.388215705016765e-01, -6.217772172617319e-01, -6.265342909947575e-01, -6.255364581651288e-01, -6.255364581651288e-01, -1.937551962122625e-01, -1.949408765669864e-01, -7.390245146921424e-01, -1.693684007243771e-01, -1.763139242862623e-01, -1.763139242862623e-01, -3.744772385434259e-03, -4.238308766550226e-03, -9.980530864197429e-02, -1.305166437166557e-03, -2.049900136428399e-03, -2.049900136428402e-03, -4.722609654849480e+00, -4.724460971773859e+00, -4.722693331516601e+00, -4.724327671314448e+00, -4.723548580209233e+00, -4.723548580209233e+00, -1.738709933144606e+00, -1.751631050227604e+00, -1.728109598207955e+00, -1.739433781877789e+00, -1.751083210105562e+00, -1.751083210105562e+00, -5.712012690789713e-01, -6.309495143144460e-01, -5.300325471303489e-01, -5.603354442975833e-01, -5.810484537493069e-01, -5.810484537493069e-01, -1.520848882541239e-01, -2.123337878904174e-01, -1.455440125390207e-01, -1.810611996459044e+00, -1.494949031234092e-01, -1.494949031234092e-01, -9.000447450714468e-04, -1.291904250837363e-03, -1.143670024952931e-03, -1.196443517724664e-01, -1.370334579868811e-03, -1.370334579868813e-03, -5.909810108651021e-01, -5.834284534607077e-01, -5.860804265848931e-01, -5.882786815227043e-01, -5.871822272144610e-01, -5.871822272144610e-01, -5.776381330243217e-01, -4.807705847135211e-01, -5.045203264400925e-01, -5.310686140360218e-01, -5.171744410420622e-01, -5.171744410420622e-01, -6.589161666593960e-01, -2.475785415878851e-01, -2.789355668448754e-01, -3.484489788471385e-01, -3.088412352829668e-01, -3.088412352829668e-01, -4.453585210284897e-01, -9.614307320006252e-02, -1.189398462551048e-01, -3.371021382859971e-01, -1.216866704435887e-01, -1.216866704435887e-01, -7.445009542535920e-03, -2.404047634999836e-04, -5.954218571641634e-04, -1.196419069305038e-01, -1.248886508955297e-03, -1.248886508955294e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mpw1lyp_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1lyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.450108744159747e-09, -5.450083745213694e-09, -5.449896526815905e-09, -5.450270043404872e-09, -5.450085137155034e-09, -5.450085137155034e-09, -7.011300835291125e-06, -7.011620481546097e-06, -7.018336326183212e-06, -7.001391570905306e-06, -7.011013144105368e-06, -7.011013144105368e-06, -3.680888649931058e-03, -3.678317683434570e-03, -3.577321697585152e-03, -3.489635653557878e-03, -3.526008875016206e-03, -3.526008875016206e-03, -1.965001442087841e-01, -2.040780857382726e-01, -2.138659008277359e-03, -4.232070781224095e-02, -1.707753123343414e-01, -1.707753123343412e-01, 2.223148673204899e+02, 2.310221249423603e+02, 3.276175250655719e+01, 2.309506188554888e+02, 2.746896477613248e+02, 2.746896477613271e+02, -1.627919373093716e-06, -1.629202086477991e-06, -1.627962108806606e-06, -1.629094624013238e-06, -1.628577646873793e-06, -1.628577646873793e-06, -5.044851667951040e-05, -4.950360831082536e-05, -5.040413123713995e-05, -4.956894423577407e-05, -4.994677161884360e-05, -4.994677161884360e-05, -7.705888684808098e-03, -7.303419307195098e-03, -9.966840964650898e-03, -1.115488324912118e-02, -7.478246605511234e-03, -7.478246605511234e-03, 1.242622456618701e+00, -5.805160314515895e-02, 1.705831431497242e+00, -9.787000956513483e-05, 2.089522608880552e-01, 2.089522608880552e-01, 3.179188589331289e+02, 2.706911025465640e+02, 1.627399146021343e+03, 7.076762371455225e+00, 7.581928483351367e+02, 7.581928483351412e+02, -1.166377457916583e-02, -1.017162403510803e-02, -1.045449878200910e-02, -1.082178515207392e-02, -1.061440516234808e-02, -1.061440516234808e-02, -1.353787848566011e-02, -1.201834167972837e-02, -1.182897627765131e-02, -1.181965863422581e-02, -1.181992632308406e-02, -1.181992632308406e-02, -6.001529576202948e-03, -6.716360645853564e-02, -6.073159811862999e-02, -4.148128572843933e-02, -5.271761525066534e-02, -5.271761525066539e-02, -1.622876222297491e-02, 3.559348425221442e+01, 1.649375926058723e+01, -5.553814950858946e-02, 2.107484220515109e+00, 2.107484220515106e+00, 1.653966776739104e+02, 1.653502107631059e+03, 6.989493351644898e+02, 2.888676393766144e+00, 9.618903390095124e+02, 9.618903390095111e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
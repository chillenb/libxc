
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_n12_sx_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12_sx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.722040752530099e-01, -1.722068492099697e-01, -1.722149013752396e-01, -1.721737198487656e-01, -1.721959063037615e-01, -1.721959063037615e-01, -9.518945623683309e-02, -9.520315857672268e-02, -9.552087997678883e-02, -9.497358580038623e-02, -9.520647339316020e-02, -9.520647339316020e-02, -3.544724042210594e-02, -3.486423452225358e-02, -1.984700481651409e-02, -2.062252621780899e-02, -2.103219886430447e-02, -2.103219886430447e-02, 3.138391502627986e-02, 2.868403262744075e-02, -4.291514798043553e-02, 4.730168491574851e-02, 4.319529958599346e-02, 4.319529958599347e-02, -1.239410412669829e-03, -1.303368690606906e-03, -1.077019754048058e-03, -6.867526603245897e-04, -8.816135250005239e-04, -8.816135250005341e-04, -1.478597971659636e-01, -1.483210796170438e-01, -1.478788949015343e-01, -1.482861418196269e-01, -1.480947267289048e-01, -1.480947267289048e-01, -2.328641782662779e-02, -2.511549590849962e-02, -1.915820318500876e-02, -2.079546371474229e-02, -2.628961642071178e-02, -2.628961642071178e-02, -6.056384150457477e-02, -9.662336710676847e-02, -5.404164285011114e-02, -8.761304584790056e-02, -6.498963869065698e-02, -6.498963869065698e-02, 3.182456315521816e-02, 4.872273326746599e-02, 2.790019997955029e-02, -1.397114395606569e-01, 4.283418772135657e-02, 4.283418772135657e-02, -5.103255971912107e-04, -6.685208631927992e-04, -4.841806276698327e-04, 1.081015549411252e-02, -6.026439720338389e-04, -6.026439720338343e-04, -1.015050402114912e-01, -9.425911583626276e-02, -9.683740287357927e-02, -9.884552989847738e-02, -9.784759173332704e-02, -9.784759173332704e-02, -1.035382091366715e-01, -3.287117305740550e-02, -5.010563157368417e-02, -7.123683955484486e-02, -6.009366964399397e-02, -6.009366964399397e-02, -9.701841509497493e-02, 3.884061105545319e-02, 1.927169389633683e-02, -2.290145733311006e-02, -2.190778431726991e-03, -2.190778431726891e-03, -3.077229629776001e-02, -1.723776316569006e-03, 2.664733784455693e-03, -3.435762635465842e-02, 2.401585838606110e-02, 2.401585838606118e-02, -1.744158738217824e-03, -1.413687666018541e-04, -3.420193972963055e-04, 2.053633342684069e-02, -5.508728384609358e-04, -5.508728384609269e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_n12_sx_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12_sx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.240602498323240e-01, -4.240605010002392e-01, -4.240616262704530e-01, -4.240578804810005e-01, -4.240598459381847e-01, -4.240598459381847e-01, -2.815827756905930e-01, -2.815887025001192e-01, -2.817271857265416e-01, -2.814996757577007e-01, -2.815916070511950e-01, -2.815916070511950e-01, -1.627019372967511e-01, -1.625355135023813e-01, -1.612624958828833e-01, -1.616131329865415e-01, -1.614402041648643e-01, -1.614402041648643e-01, -7.505403536213352e-02, -8.094308810326967e-02, -1.731173045956867e-01, 2.803200735261838e-02, -2.331272872625414e-02, -2.331272872625408e-02, -1.538998301984325e-03, -1.594091413408089e-03, 1.344730159610680e-02, -9.205307515839798e-04, -1.148778018042936e-03, -1.148778018043003e-03, -3.108953831477682e-01, -3.105941672050814e-01, -3.108832483972481e-01, -3.106174604017843e-01, -3.107430747250139e-01, -3.107430747250139e-01, -2.304844761865079e-01, -2.305484745046490e-01, -2.311839649231311e-01, -2.311836422258691e-01, -2.302483225684464e-01, -2.302483225684464e-01, -1.655637315020605e-01, -1.422851497778011e-01, -1.590429740473079e-01, -1.469476270005847e-01, -1.675412461135459e-01, -1.675412461135459e-01, 8.967771094893631e-02, -1.538730804288198e-02, 8.672109526081938e-02, -1.984534020770092e-01, 6.795690988500805e-02, 6.795690988500805e-02, -6.931915909781899e-04, -8.951818671343819e-04, -6.382537253436804e-04, 5.405102218508949e-02, -7.944505399538360e-04, -7.944505399538858e-04, -1.266567221878122e-01, -1.366666902756864e-01, -1.307470493279098e-01, -1.274435467487634e-01, -1.288387792232987e-01, -1.288387792232987e-01, -1.258728873780084e-01, -1.454336085854835e-01, -1.546792650499539e-01, -1.607495714529488e-01, -1.596661670472518e-01, -1.596661670472518e-01, -1.477971374246655e-01, -7.613799944494810e-02, -1.141681144233783e-01, -1.259667264168356e-01, -1.225490953958909e-01, -1.225490953958912e-01, -1.406573555808612e-01, 1.063870095408208e-02, 2.924512045841494e-02, -1.268506178554103e-01, 7.888973464215600e-02, 7.888973464215535e-02, -1.984034273240938e-03, -1.968565510932803e-04, -4.687498254620017e-04, 7.417017464679931e-02, -7.279044498343333e-04, -7.279044498343738e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_n12_sx_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12_sx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.108017104670303e-10, 6.108045086393619e-10, 6.107967785008299e-10, 6.107555560261055e-10, 6.107800276763600e-10, 6.107800276763600e-10, 2.852842519509484e-06, 2.853250781690590e-06, 2.862383509920660e-06, 2.843978248703898e-06, 2.853006141418450e-06, 2.853006141418450e-06, 4.389148079289537e-03, 4.376654001960766e-03, 4.172040635588798e-03, 4.043603651946916e-03, 4.093767014790538e-03, 4.093767014790538e-03, 8.273471472476341e-01, 8.423330506684519e-01, 2.325829740994689e-03, 3.735089572328513e-01, 8.019282137690495e-01, 8.019282137690498e-01, -2.393032054720365e+01, -2.512653654473459e+01, -9.020034072959394e+00, -2.291985340524522e+01, -2.854982528009706e+01, -2.854982528012318e+01, 6.119064888821111e-07, 6.130224254826174e-07, 6.119485422757465e-07, 6.129342297283148e-07, 6.124807366119405e-07, 6.124807366119405e-07, 2.375883736757878e-05, 2.323335284721538e-05, 2.373711227809076e-05, 2.326062779284440e-05, 2.349236279918381e-05, 2.349236279918381e-05, 1.326730398232958e-02, 1.127799867392573e-02, 1.725859259219478e-02, 2.159577282798112e-02, 1.328779975199206e-02, 1.328779975199206e-02, -1.746510188949018e+00, 3.271852031004083e-01, -2.331762556204778e+00, 6.441703687413615e-05, -4.484146227477973e-01, -4.484146227477973e-01, -3.076743881489217e+01, -2.701915401524461e+01, -1.727065476698446e+02, -6.540382518763534e+00, -7.937639924014169e+01, -7.937639924034738e+01, 1.805158976722685e-02, 1.631989909481604e-02, 1.488987610890901e-02, 1.451544493091328e-02, 1.451890853089754e-02, 1.451890853089754e-02, 3.275476428345592e-02, 1.869002190959601e-02, 2.063369862210641e-02, 2.459425464667700e-02, 2.264608296131182e-02, 2.264608296131182e-02, 9.354634218309369e-03, 2.551306929173483e-01, 1.796109678437118e-01, 8.389217170525998e-02, 1.244393955300547e-01, 1.244393955300547e-01, 2.687449891042712e-02, -8.478933883896499e+00, -6.898821090325539e+00, 1.242511274384293e-01, -4.383332274380822e+00, -4.383332274380781e+00, -1.845401367211796e+01, -1.420272515324610e+02, -6.659766919408447e+01, -5.151634046626531e+00, -1.008085829515996e+02, -1.008085829516168e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_htbs_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_htbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.084480773808810e+01, -2.084484004646048e+01, -2.084504305211282e+01, -2.084456114126591e+01, -2.084480521406141e+01, -2.084480521406141e+01, -3.445209449168930e+00, -3.445180805931457e+00, -3.444602846398097e+00, -3.446261722368234e+00, -3.445258392387285e+00, -3.445258392387285e+00, -6.855294505029607e-01, -6.853668308091878e-01, -6.829869558762214e-01, -6.878101192490975e-01, -6.893525722435372e-01, -6.893525722435372e-01, -2.173403477910039e-01, -2.168279162785645e-01, -7.936227363905506e-01, -2.002062934979947e-01, -2.149258399702005e-01, -2.149258399702004e-01, -1.011217354967865e-02, -1.064899078704721e-02, -5.870031516958291e-02, -5.830218135953091e-03, -8.138160784503223e-03, -8.138160784503223e-03, -5.020898987147882e+00, -5.020480279145280e+00, -5.020889548089352e+00, -5.020519768160849e+00, -5.020681472693921e+00, -5.020681472693921e+00, -2.060672760359135e+00, -2.071258840691894e+00, -2.059867514452637e+00, -2.069138008142796e+00, -2.067025836799761e+00, -2.067025836799761e+00, -5.770105698493307e-01, -6.024454532402853e-01, -5.376801419306901e-01, -5.370072854351768e-01, -5.840378065612007e-01, -5.840378065612009e-01, -1.515850083061948e-01, -2.480691445525315e-01, -1.408938867254587e-01, -1.812601157650399e+00, -1.715069022416197e-01, -1.715069022416197e-01, -4.501092174754964e-03, -5.699392574836056e-03, -4.364675792400219e-03, -9.509222078884308e-02, -5.482775671063140e-03, -5.482775671063142e-03, -5.507240987769813e-01, -5.535987598660888e-01, -5.526310942876247e-01, -5.517906755729649e-01, -5.522142623346302e-01, -5.522142623346302e-01, -5.339774319320572e-01, -5.021937881506913e-01, -5.124610254418585e-01, -5.214395543694568e-01, -5.168379213606948e-01, -5.168379213606948e-01, -6.326434745509135e-01, -2.808494973339598e-01, -3.038939695226663e-01, -3.586018791549147e-01, -3.262298177940329e-01, -3.262298177940327e-01, -4.632141005946112e-01, -5.610195407095957e-02, -7.652561080908953e-02, -3.378360409983593e-01, -1.208208161380267e-01, -1.208208161380267e-01, -1.425425101214373e-02, -1.523316300276004e-03, -3.203544973302094e-03, -1.135474253874175e-01, -5.034982857663592e-03, -5.034982857663587e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_htbs_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_htbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.547286458173648e+01, -2.547283314737357e+01, -2.547296401761203e+01, -2.547290537417099e+01, -2.547327545347374e+01, -2.547335351033131e+01, -2.547215268781370e+01, -2.547192239788060e+01, -2.547292422466695e+01, -2.547248953112007e+01, -2.547292422466695e+01, -2.547248953112007e+01, -4.122388191578684e+00, -4.122424759861231e+00, -4.122412696895897e+00, -4.122447209280507e+00, -4.122987024652704e+00, -4.123161890694417e+00, -4.122657891970539e+00, -4.122817572726860e+00, -4.121662309087395e+00, -4.123405387744225e+00, -4.121662309087395e+00, -4.123405387744225e+00, -7.845488738980102e-01, -7.889788399985819e-01, -7.828392300814326e-01, -7.883175206198512e-01, -7.576314183995162e-01, -7.474624830622570e-01, -7.581620943138468e-01, -7.613952795070871e-01, -7.967532234913739e-01, -6.885823725151536e-01, -7.967532234913739e-01, -6.885823725151536e-01, -1.338622721920654e-01, -1.442113416305609e-01, -1.381038140687596e-01, -1.518650822170319e-01, -9.020635381222493e-01, -9.373201074023217e-01, -1.751390836963614e-01, -1.740551704327926e-01, -1.408620818234060e-01, -1.824642814817505e-01, -1.408620818234063e-01, -1.824642814817504e-01, -1.303761629899009e-02, -1.385400848819942e-02, -1.366186325065716e-02, -1.463530105325926e-02, -7.596593310481209e-02, -8.022113756772861e-02, -7.837274757367561e-03, -7.706682731293256e-03, -1.163080014436929e-02, -6.618657626552290e-03, -1.163080014436929e-02, -6.618657626552290e-03, -6.249305254221553e+00, -6.247767600988702e+00, -6.251477829931392e+00, -6.249866190673338e+00, -6.249427617523177e+00, -6.247840924269771e+00, -6.251291716430463e+00, -6.249747323357010e+00, -6.250416113055888e+00, -6.248820863242670e+00, -6.250416113055888e+00, -6.248820863242670e+00, -2.245958757958610e+00, -2.245831033540206e+00, -2.267628591347211e+00, -2.266893664276154e+00, -2.216999321436374e+00, -2.225542578104391e+00, -2.236561033075972e+00, -2.244996177143410e+00, -2.277807992029147e+00, -2.259748112937871e+00, -2.277807992029147e+00, -2.259748112937871e+00, -6.983014369467810e-01, -6.964894022834944e-01, -7.754108334924659e-01, -7.760790382846815e-01, -6.314389958143424e-01, -6.556897264177959e-01, -6.723870463073092e-01, -6.944017351751894e-01, -7.281484716504959e-01, -6.912503962653725e-01, -7.281484716504960e-01, -6.912503962653727e-01, -1.886517654210109e-01, -1.888783344267085e-01, -1.634767509332364e-01, -1.626131921303760e-01, -1.764583214444982e-01, -1.823325752498242e-01, -2.338672499448301e+00, -2.337672021507737e+00, -1.880457810108493e-01, -1.734933942786387e-01, -1.880457810108493e-01, -1.734933942786387e-01, -5.879441569503017e-03, -6.110164258163266e-03, -7.540839044572501e-03, -7.654969939076097e-03, -5.638837180345527e-03, -5.971727633039896e-03, -1.262773510124046e-01, -1.270608154912789e-01, -5.759214710703059e-03, -7.909246535848055e-03, -5.759214710703061e-03, -7.909246535848059e-03, -7.234772506714284e-01, -7.264024292392502e-01, -7.124248804662557e-01, -7.153962580775045e-01, -7.161306562412076e-01, -7.191128519844725e-01, -7.193627515905613e-01, -7.222840821012056e-01, -7.177306303596399e-01, -7.206819169874178e-01, -7.177306303596399e-01, -7.206819169874178e-01, -7.063932717957138e-01, -7.087472569390746e-01, -5.767274578047967e-01, -5.794483274888523e-01, -6.091314041520268e-01, -6.120280805969818e-01, -6.438240001432545e-01, -6.462024927029079e-01, -6.256969712481617e-01, -6.281172981104896e-01, -6.256969712481617e-01, -6.281172981104896e-01, -8.112726950582441e-01, -8.130901151254791e-01, -1.710794971858771e-01, -1.725138661924342e-01, -2.474101519897805e-01, -2.540369686170927e-01, -4.046148461878450e-01, -4.070612643982504e-01, -3.348593399996875e-01, -3.353057524048301e-01, -3.348593399996876e-01, -3.353057524048297e-01, -5.297454592263204e-01, -5.338101426443952e-01, -7.454496921854471e-02, -7.505502475361867e-02, -1.002729502485271e-01, -1.036294097232134e-01, -3.919769882582131e-01, -3.982197857199115e-01, -1.536662129478288e-01, -1.571551667534733e-01, -1.536662129478288e-01, -1.571551667534734e-01, -1.864956171239904e-02, -1.932569049950109e-02, -2.028762023808494e-03, -2.033398898287572e-03, -4.128061189754679e-03, -4.390526307869538e-03, -1.477187576369757e-01, -1.497873070984669e-01, -5.455199472438233e-03, -7.249407432700401e-03, -5.455199472438227e-03, -7.249407432700393e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_htbs_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_htbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.174886394407835e-08, 0.000000000000000e+00, -1.174893365571410e-08, -1.174882737721662e-08, 0.000000000000000e+00, -1.174890702351738e-08, -1.174852628869013e-08, 0.000000000000000e+00, -1.174852713098713e-08, -1.174894394560769e-08, 0.000000000000000e+00, -1.174905798253001e-08, -1.174885143172931e-08, 0.000000000000000e+00, -1.174871035638451e-08, -1.174885143172931e-08, 0.000000000000000e+00, -1.174871035638451e-08, -1.507863288681665e-05, 0.000000000000000e+00, -1.508628986681406e-05, -1.507929977501675e-05, 0.000000000000000e+00, -1.508750937173358e-05, -1.510107676060267e-05, 0.000000000000000e+00, -1.510644549416945e-05, -1.505364398009055e-05, 0.000000000000000e+00, -1.506029794529315e-05, -1.508602840474176e-05, 0.000000000000000e+00, -1.507824457183330e-05, -1.508602840474176e-05, 0.000000000000000e+00, -1.507824457183330e-05, -9.073587727999596e-03, 0.000000000000000e+00, -9.082441364817574e-03, -9.081378864176518e-03, 0.000000000000000e+00, -9.087789060773005e-03, -9.389843006371698e-03, 0.000000000000000e+00, -9.569931527363872e-03, -9.204531476902150e-03, 0.000000000000000e+00, -9.154387833914741e-03, -9.058504256604776e-03, 0.000000000000000e+00, -1.088296315481433e-02, -9.058504256604776e-03, 0.000000000000000e+00, -1.088296315481433e-02, -2.389481874740738e+00, 0.000000000000000e+00, -2.091765474142051e+00, -2.307342149003843e+00, 0.000000000000000e+00, -1.954306169079885e+00, -5.256138053679088e-03, 0.000000000000000e+00, -4.965594447862699e-03, -2.514825274634430e+00, 0.000000000000000e+00, -2.495068247819836e+00, -2.024826456799515e+00, 0.000000000000000e+00, -1.801459099249745e+00, -2.024826456799510e+00, 0.000000000000000e+00, -1.801459099249751e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.773605429518733e-292, -2.693936877908569e-291, 0.000000000000000e+00, -7.765252845059571e-254, -8.237879739827784e-10, 0.000000000000000e+00, -1.536227976768569e-08, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.606609779439472e-06, 0.000000000000000e+00, -3.609799680033344e-06, -3.610507487690617e-06, 0.000000000000000e+00, -3.613571162790701e-06, -3.606782803269968e-06, 0.000000000000000e+00, -3.609898118312190e-06, -3.610127785802669e-06, 0.000000000000000e+00, -3.613324752275700e-06, -3.608640344641787e-06, 0.000000000000000e+00, -3.611699965765431e-06, -3.608640344641787e-06, 0.000000000000000e+00, -3.611699965765431e-06, -1.162437412447939e-04, 0.000000000000000e+00, -1.162668487687271e-04, -1.130505819053220e-04, 0.000000000000000e+00, -1.131690362962867e-04, -1.189823506965738e-04, 0.000000000000000e+00, -1.181717769983423e-04, -1.159937129002502e-04, 0.000000000000000e+00, -1.152092513693141e-04, -1.127904551016299e-04, 0.000000000000000e+00, -1.143820632554566e-04, -1.127904551016299e-04, 0.000000000000000e+00, -1.143820632554566e-04, -1.946420222028643e-02, 0.000000000000000e+00, -1.968973473806945e-02, -1.832781518444631e-02, 0.000000000000000e+00, -1.835435155080312e-02, -2.706937418096195e-02, 0.000000000000000e+00, -2.417815765156881e-02, -3.071973247900855e-02, 0.000000000000000e+00, -2.684280404390718e-02, -1.735208099639541e-02, 0.000000000000000e+00, -2.100078617812710e-02, -1.735208099639541e-02, 0.000000000000000e+00, -2.100078617812710e-02, -8.550001897762636e-01, 0.000000000000000e+00, -9.414280400943430e-01, -1.526620897761764e+00, 0.000000000000000e+00, -1.518755586011082e+00, -5.990103108800141e-01, 0.000000000000000e+00, -7.976880443128630e-01, -2.241327844819293e-04, 0.000000000000000e+00, -2.245740416252768e-04, -1.924834665564391e+00, 0.000000000000000e+00, -3.001130223152962e+00, -1.924834665564391e+00, 0.000000000000000e+00, -3.001130223152962e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.886467143232868e-02, 0.000000000000000e+00, -3.559504590584001e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.719909332522027e-02, 0.000000000000000e+00, -2.681438344309263e-02, -2.595065574068391e-02, 0.000000000000000e+00, -2.560284905172735e-02, -2.637415372063499e-02, 0.000000000000000e+00, -2.601675166135449e-02, -2.674098729766360e-02, 0.000000000000000e+00, -2.636737158905680e-02, -2.655647964618523e-02, 0.000000000000000e+00, -2.619084095299321e-02, -2.655647964618523e-02, 0.000000000000000e+00, -2.619084095299321e-02, -3.099002238650566e-02, 0.000000000000000e+00, -3.058882650891134e-02, -3.175957697109513e-02, 0.000000000000000e+00, -3.135080894416774e-02, -3.071251787553729e-02, 0.000000000000000e+00, -3.037206583635785e-02, -3.087375838994137e-02, 0.000000000000000e+00, -3.050976670249249e-02, -3.082497417426893e-02, 0.000000000000000e+00, -3.044732977278987e-02, -3.082497417426893e-02, 0.000000000000000e+00, -3.044732977278987e-02, -1.503112244045382e-02, 0.000000000000000e+00, -1.500857584699954e-02, -8.626084202540649e-01, 0.000000000000000e+00, -8.469470546463929e-01, -4.170964388676409e-01, 0.000000000000000e+00, -3.970807866534138e-01, -1.225555473784451e-01, 0.000000000000000e+00, -1.202178360761914e-01, -2.070111204521554e-01, 0.000000000000000e+00, -2.070127443751878e-01, -2.070111204521552e-01, 0.000000000000000e+00, -2.070127443751874e-01, -4.397682812859512e-02, 0.000000000000000e+00, -4.312718017898971e-02, -6.933362049982141e-11, 0.000000000000000e+00, -1.064319239518008e-10, -3.429533441575585e-05, 0.000000000000000e+00, -1.129104052683015e-04, -1.600928545439704e-01, 0.000000000000000e+00, -1.531186151698788e-01, -5.379466090379250e-01, 0.000000000000000e+00, -1.310446239359532e+00, -5.379466090379268e-01, 0.000000000000000e+00, -1.310446239359535e+00, -1.817570895455463e-178, 0.000000000000000e+00, -4.058635175596174e-164, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.211345687584151e-01, 0.000000000000000e+00, -5.644225227149000e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
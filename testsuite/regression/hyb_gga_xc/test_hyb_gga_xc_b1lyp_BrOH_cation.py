
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b1lyp_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.585939265880253e+01, -1.585941285046794e+01, -1.585955449612464e+01, -1.585925301565779e+01, -1.585940361027063e+01, -1.585940361027063e+01, -2.677378750217910e+00, -2.677350909350105e+00, -2.676771611025182e+00, -2.678277494277805e+00, -2.677408843159431e+00, -2.677408843159431e+00, -5.621084621170545e-01, -5.619940942527646e-01, -5.601558170696159e-01, -5.640245753765682e-01, -5.630461653361000e-01, -5.630461653361000e-01, -1.680897993136627e-01, -1.697087254065106e-01, -6.467574116152932e-01, -1.366480319440758e-01, -1.619592798965709e-01, -1.619592798965709e-01, -4.973945222630737e-02, -4.978535021615739e-02, -8.490218340862278e-02, -4.531882816084228e-02, -4.575675808077216e-02, -4.575675808077213e-02, -3.856412005108956e+00, -3.855907772323475e+00, -3.856397115889320e+00, -3.855951851573550e+00, -3.856151897103215e+00, -3.856151897103215e+00, -1.639540677637954e+00, -1.647311902801416e+00, -1.639864706656178e+00, -1.646728728842807e+00, -1.643685743756174e+00, -1.643685743756174e+00, -4.761676541103281e-01, -4.961383806107034e-01, -4.450604073879989e-01, -4.448969478452472e-01, -4.815889483647373e-01, -4.815889483647373e-01, -1.062590595757455e-01, -1.734245079195136e-01, -1.019675997921176e-01, -1.420981673898289e+00, -1.172812555727372e-01, -1.172812555727372e-01, -4.086575262232883e-02, -4.366565424297655e-02, -2.862015449457118e-02, -8.827162483695770e-02, -3.477988254475186e-02, -3.477988254475187e-02, -4.556998087368442e-01, -4.579826358621105e-01, -4.572274392894158e-01, -4.565618589423273e-01, -4.568986035597730e-01, -4.568986035597730e-01, -4.425686345021663e-01, -4.166884085024680e-01, -4.251316905811912e-01, -4.325176850454119e-01, -4.287311025871318e-01, -4.287311025871318e-01, -5.196699115956886e-01, -2.120261218654603e-01, -2.455293068323322e-01, -3.009733723890029e-01, -2.717268422475336e-01, -2.717268422475337e-01, -3.854992001754217e-01, -8.546147903351090e-02, -8.768021624214517e-02, -2.849964230270310e-01, -9.397423600307417e-02, -9.397423600307417e-02, -5.617475315292319e-02, -2.517169745493095e-02, -3.297547452009910e-02, -9.177393086327384e-02, -3.270343070608860e-02, -3.270343070608858e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b1lyp_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.898099634179367e+01, -1.898097388976036e+01, -1.898107840526458e+01, -1.898103378562103e+01, -1.898132607235948e+01, -1.898139125702520e+01, -1.898039590543476e+01, -1.898020990454642e+01, -1.898104314925597e+01, -1.898067273622498e+01, -1.898104314925597e+01, -1.898067273622498e+01, -3.115041961633728e+00, -3.115115577861223e+00, -3.115069328754842e+00, -3.115144608863734e+00, -3.115750409245121e+00, -3.115917729137098e+00, -3.115067547261504e+00, -3.115230883722782e+00, -3.114532628512718e+00, -3.115839207682373e+00, -3.114532628512718e+00, -3.115839207682373e+00, -6.377778431367336e-01, -6.392415946775757e-01, -6.368485313970114e-01, -6.386200888043034e-01, -6.206150205404114e-01, -6.190599782286631e-01, -6.243564803223892e-01, -6.248485085488151e-01, -6.339858264483857e-01, -6.194997018518035e-01, -6.339858264483857e-01, -6.194997018518035e-01, -1.899866624720683e-01, -1.900970306647878e-01, -1.914026127282439e-01, -1.917411560637202e-01, -7.296454547627569e-01, -7.466235212533785e-01, -1.594396160973626e-01, -1.598349594104220e-01, -1.595822729334802e-01, -1.736831750440803e-01, -1.595822729334801e-01, -1.736831750440803e-01, -1.482969567991392e-02, -1.488022132841058e-02, -1.516850349491216e-02, -1.520900895391298e-02, -4.480238249977226e-02, -4.547099359144992e-02, -1.130507707133552e-02, -1.132796679225345e-02, -1.272595454888734e-02, -1.078950789785052e-02, -1.272595454888732e-02, -1.078950789785051e-02, -4.723441586159525e+00, -4.722332333162544e+00, -4.725336844441685e+00, -4.724167269527348e+00, -4.723545687614462e+00, -4.722396722006920e+00, -4.725174927514399e+00, -4.724060524113789e+00, -4.724411418555112e+00, -4.723254437328281e+00, -4.724411418555112e+00, -4.723254437328281e+00, -1.732083368805096e+00, -1.732011815709550e+00, -1.745407676207783e+00, -1.745015027226264e+00, -1.719555830330593e+00, -1.722031316565949e+00, -1.731030606142790e+00, -1.733664554739863e+00, -1.748225647740469e+00, -1.741509651902809e+00, -1.748225647740469e+00, -1.741509651902809e+00, -5.715340475057267e-01, -5.705676068975153e-01, -6.308682484406652e-01, -6.312429096380794e-01, -5.235850647960721e-01, -5.358719907715159e-01, -5.543210298705630e-01, -5.663246685229205e-01, -5.905666652727398e-01, -5.708716298164995e-01, -5.905666652727398e-01, -5.708716298164996e-01, -1.258243890125375e-01, -1.255255353270002e-01, -2.042185532110616e-01, -2.042249254944689e-01, -1.153235464212999e-01, -1.167909425768826e-01, -1.811294878145579e+00, -1.810616389988981e+00, -1.380394197348931e-01, -1.331997782719174e-01, -1.380394197348931e-01, -1.331997782719174e-01, -9.535115467630955e-03, -9.761209114845884e-03, -1.089708380706353e-02, -1.102391978547777e-02, -7.654528931198106e-03, -7.586101413951791e-03, -7.374974698300113e-02, -7.432826737291340e-02, -9.025285873406313e-03, -9.298159854734422e-03, -9.025285873406336e-03, -9.298159854734422e-03, -5.908125925533054e-01, -5.924619227318889e-01, -5.827065159389129e-01, -5.843549816140834e-01, -5.854119717975507e-01, -5.870756788549474e-01, -5.877728413723198e-01, -5.894099986463482e-01, -5.865790426254963e-01, -5.882293146177520e-01, -5.865790426254963e-01, -5.882293146177520e-01, -5.775300661581350e-01, -5.788484262765097e-01, -4.794585921315304e-01, -4.806065129665577e-01, -5.035465377266494e-01, -5.049600124363510e-01, -5.304487867019083e-01, -5.316980668563749e-01, -5.164483354485884e-01, -5.176874674978005e-01, -5.164483354485884e-01, -5.176874674978005e-01, -6.585001684375801e-01, -6.595331744744415e-01, -2.422272401057638e-01, -2.422094184938570e-01, -2.758105788572169e-01, -2.759239835028114e-01, -3.472862710851403e-01, -3.481564300388270e-01, -3.073264210651627e-01, -3.071479349210754e-01, -3.073264210651628e-01, -3.071479349210754e-01, -4.438407381604096e-01, -4.454316775528815e-01, -4.379447579629412e-02, -4.386692718322113e-02, -5.870770079145889e-02, -5.951230993036763e-02, -3.355290970118619e-01, -3.379896837227463e-01, -9.435591937352492e-02, -9.319358336327552e-02, -9.435591937352489e-02, -9.319358336327550e-02, -1.849516375464929e-02, -1.845886041330970e-02, -5.154510221679033e-03, -4.721600963389763e-03, -7.470693029272886e-03, -7.459032114191807e-03, -8.778576588859099e-02, -8.843473468344532e-02, -7.978474118837916e-03, -8.897828978247204e-03, -7.978474118837927e-03, -8.897828978247188e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b1lyp_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.086366635328540e-08, 1.463652377038497e-11, -1.086372766967848e-08, -1.086360586738464e-08, 1.463631512328463e-11, -1.086368312590560e-08, -1.086327570254595e-08, 1.463532316351641e-11, -1.086325158605616e-08, -1.086396951838130e-08, 1.463842899785011e-11, -1.086412653742327e-08, -1.086364102632627e-08, 1.463681083685537e-11, -1.086367489538826e-08, -1.086364102632627e-08, 1.463681083685537e-11, -1.086367489538826e-08, -1.410073775228069e-05, 1.902747638536065e-07, -1.410605937829089e-05, -1.410114739080926e-05, 1.902725761814505e-07, -1.410686219849813e-05, -1.411494211355906e-05, 1.901971754504621e-07, -1.411837594755705e-05, -1.408184693708801e-05, 1.901358235030523e-07, -1.408622182988644e-05, -1.410723390698657e-05, 1.902491360705296e-07, -1.409831186998755e-05, -1.410723390698657e-05, 1.902491360705296e-07, -1.409831186998755e-05, -7.953372162617467e-03, 1.144533678933034e-03, -8.023021077200210e-03, -7.945364433696213e-03, 1.151156568500561e-03, -8.030768078032893e-03, -7.986904431667267e-03, 1.320750241138922e-03, -7.864388683703790e-03, -7.697415357467084e-03, 1.265386257803329e-03, -7.738637501480562e-03, -8.355811774341112e-03, 1.376562597771172e-03, -7.059247256451227e-03, -8.355811774341112e-03, 1.376562597771172e-03, -7.059247256451227e-03, -7.653151519829424e-01, 6.012876550391961e-01, -7.625109731434830e-01, -7.585707378253999e-01, 5.680272535784819e-01, -7.508639919767537e-01, -4.597126616318697e-03, 5.322727974273029e-04, -4.517328532680829e-03, -1.293980839471364e+00, 1.675901373768304e+00, -1.283222745836888e+00, -1.077374512328929e+00, 2.382014110698246e+00, -3.733321917750373e+00, -1.077374512328928e+00, 2.382014110698246e+00, -3.733321917750378e+00, -1.667951090267195e+04, 1.111914508308031e-07, -1.397216243180049e+04, -1.480198937423108e+04, 4.562559950974850e-07, -1.213307090083609e+04, -8.553459092874994e+01, 1.838627425311693e+01, -7.344562854199420e+01, -7.031809866838512e+04, 4.136895542502456e-17, -7.323966409429624e+04, -2.313986626477509e+04, 1.070835888305008e-12, -1.521050847405917e+05, -2.313986626477512e+04, 1.070835888305008e-12, -1.521050847405916e+05, -3.244955349638160e-06, 2.233703107768369e-08, -3.247675591651097e-06, -3.247371683608830e-06, 2.231537777319893e-08, -3.250008777108965e-06, -3.245056700667475e-06, 2.233595490630417e-08, -3.247729471785345e-06, -3.247126793822513e-06, 2.231683364972891e-08, -3.249852909070813e-06, -3.246219123800955e-06, 2.232609670798167e-08, -3.248850610032443e-06, -3.246219123800955e-06, 2.232609670798167e-08, -3.248850610032443e-06, -1.044592968747232e-04, 4.104557134069705e-06, -1.044733789431970e-04, -1.023733149910066e-04, 3.949027730077658e-06, -1.024398841405042e-04, -1.045089811229027e-04, 4.223201907834492e-06, -1.046644096181874e-04, -1.027016191849512e-04, 4.080707689470805e-06, -1.028141925062109e-04, -1.033199511039834e-04, 3.966397450084659e-06, -1.032178364861878e-04, -1.033199511039834e-04, 3.966397450084659e-06, -1.032178364861878e-04, -1.642548888722344e-02, 2.253646609880290e-03, -1.655947028768828e-02, -1.515439590282823e-02, 1.482953633834817e-03, -1.518843026981799e-02, -2.241922430424580e-02, 3.393155540240489e-03, -2.083511099850620e-02, -2.474382858169439e-02, 2.786872153422916e-03, -2.233599192097524e-02, -1.498562349441577e-02, 2.171399762139616e-03, -1.713320114614795e-02, -1.498562349441578e-02, 2.171399762139616e-03, -1.713320114614795e-02, -2.773413128276341e+00, 5.364331359216309e+00, -2.818460252826376e+00, -5.118931092413903e-01, 5.928062603673171e-01, -5.133638061266874e-01, -3.754104965368703e+00, 6.670630325612285e+00, -3.635417725750537e+00, -1.955714114119551e-04, 4.550774239056777e-06, -1.959145057472272e-04, -2.091319302246485e+00, 3.362374819306393e+00, -2.454806892540555e+00, -2.091319302246485e+00, 3.362374819306393e+00, -2.454806892540555e+00, -1.770219885045288e+05, 5.421496698436063e-24, -1.521662757871417e+05, -8.286111177391569e+04, 1.228758886113335e-17, -7.759902392012546e+04, -3.115331090749321e+05, 5.656440620065503e-25, -2.721602011624266e+05, -1.632814910217796e+01, 1.504669301565768e+01, -1.581085930767728e+01, -2.410382459986042e+05, 9.610322787727072e-20, -9.625509169745266e+04, -2.410382459986039e+05, 9.610322787727070e-20, -9.625509169745262e+04, -2.281777914954397e-02, 2.199677179191681e-03, -2.258428035345992e-02, -2.131787676973897e-02, 2.264286130145305e-03, -2.111595996857018e-02, -2.176330041210549e-02, 2.241303100651475e-03, -2.155496130223029e-02, -2.219438436217867e-02, 2.222568983976134e-03, -2.197008774779702e-02, -2.197143105991089e-02, 2.231922961512953e-03, -2.175521357149187e-02, -2.197143105991089e-02, 2.231922961512953e-03, -2.175521357149187e-02, -2.633405804840581e-02, 2.525278278664052e-03, -2.606885773567957e-02, -2.698913806916898e-02, 5.397863529116129e-03, -2.683025666319017e-02, -2.588179338788766e-02, 4.298036410666330e-03, -2.571317128840350e-02, -2.531546526996697e-02, 3.458400931093935e-03, -2.510927723023262e-02, -2.558910288893345e-02, 3.860241852276062e-03, -2.537884109501213e-02, -2.558910288893345e-02, 3.860241852276062e-03, -2.537884109501213e-02, -1.244741055332289e-02, 1.166454564355177e-03, -1.245256406650425e-02, -2.771231070809457e-01, 2.273047300256539e-01, -2.779666797358810e-01, -1.823294885514277e-01, 1.027440095908986e-01, -1.831132078507036e-01, -9.881789585788484e-02, 2.883676467546989e-02, -9.795836211835762e-02, -1.362192134276982e-01, 5.500747727544229e-02, -1.368961451250833e-01, -1.362192134276982e-01, 5.500747727544231e-02, -1.368961451250834e-01, -3.690374361487740e-02, 8.100714365057125e-03, -3.662660053021715e-02, -8.897287014755206e+01, 1.766533536733120e+01, -8.729927380044647e+01, -3.231569355043036e+01, 1.878803243082290e+01, -2.987013367834748e+01, -1.302086455781870e-01, 3.428914672188251e-02, -1.270591933028205e-01, -7.764550345424825e+00, 9.701438036301546e+00, -7.909705443876478e+00, -7.764550345424825e+00, 9.701438036301546e+00, -7.909705443876486e+00, -5.596974223239731e+03, 3.417478728614279e-04, -5.084531202681107e+03, -5.233292617492661e+06, 2.850114531961813e-85, -6.089592083944468e+06, -5.872283387040542e+05, 1.662312648545182e-36, -5.001555456855020e+05, -9.724848699778995e+00, 1.102889579754806e+01, -9.357424116526204e+00, -3.450847710281463e+05, 8.178814636871498e-22, -1.249632386224339e+05, -3.450847710281471e+05, 8.178814636871070e-22, -1.249632386224343e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
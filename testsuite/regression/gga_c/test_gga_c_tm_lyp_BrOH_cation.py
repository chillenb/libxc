
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_tm_lyp_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.853301649495073e-02, -5.853313019462097e-02, -5.853345751586305e-02, -5.853176947310578e-02, -5.853267929708814e-02, -5.853267929708814e-02, -5.308209177658089e-02, -5.308327994468186e-02, -5.311083497674207e-02, -5.306375586641796e-02, -5.308360428442878e-02, -5.308360428442878e-02, -3.468746153330828e-02, -3.453477954028490e-02, -3.027913439743970e-02, -3.059831615886147e-02, -2.920418483537948e-02, -2.920418483537948e-02, -5.033596248881304e-03, -6.166536051184197e-03, -3.762446057123725e-02, 5.091433525701079e-03, 3.349055425452427e-05, 3.349055425454561e-05, -1.873828909934996e-03, -1.962692028891963e-03, -8.205957494856303e-03, -1.113117362790503e-03, -7.302817011353914e-04, -7.302817011353915e-04, -5.724998041755328e-02, -5.727395529981957e-02, -5.725097555289661e-02, -5.727214421897452e-02, -5.726219776225448e-02, -5.726219776225448e-02, -4.090642605411703e-02, -4.131030514523250e-02, -4.004902907562249e-02, -4.042174645912881e-02, -4.152421291488637e-02, -4.152421291488637e-02, -3.836451018801935e-02, -4.354385289620473e-02, -3.652226603781625e-02, -4.132728610834845e-02, -3.912251073726298e-02, -3.912251073726298e-02, 7.649810003766373e-03, 7.641695118882611e-03, 5.240083741719052e-03, -5.447274668661357e-02, 4.575529577274542e-03, 4.575529577274542e-03, -8.621127636061072e-04, -1.089001221458868e-03, -8.324201069401857e-04, -5.473517583916261e-03, -8.085369335722426e-04, -8.085369335722425e-04, -4.349870198601601e-02, -4.250035377717733e-02, -4.285337703717991e-02, -4.314397797332725e-02, -4.299885968489843e-02, -4.299885968489843e-02, -4.339245519790009e-02, -3.118975643646031e-02, -3.536914179297677e-02, -3.898820949442866e-02, -3.724289963123700e-02, -3.724289963123700e-02, -4.400720114600823e-02, 9.758197401685864e-04, -8.986565141931474e-03, -2.550546863002618e-02, -1.793240812378176e-02, -1.793240812378177e-02, -2.989738891201070e-02, -8.118053555700866e-03, -7.530849529976034e-03, -2.813121088474872e-02, -2.081196147317558e-03, -2.081196147317560e-03, -2.607403628107283e-03, -2.972801216717641e-04, -6.136055821946612e-04, -3.164713400348732e-03, -7.815059176373896e-04, -7.815059176373891e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_tm_lyp_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.881419186905521e-02, -6.881426722621117e-02, -6.881382144594415e-02, -6.881426573692170e-02, -6.881413427284203e-02, -6.881289914824638e-02, -6.881480596257535e-02, -6.881774875072430e-02, -6.881093775282798e-02, -6.881863878885737e-02, -6.881093775282798e-02, -6.881863878885737e-02, -6.998981444335520e-02, -6.994645192820211e-02, -6.998912030134884e-02, -6.994314662118875e-02, -6.994316789374802e-02, -6.989680702366882e-02, -7.002643937144222e-02, -6.997471985440236e-02, -7.004020949590800e-02, -6.989147280447372e-02, -7.004020949590800e-02, -6.989147280447372e-02, -6.943321343928786e-02, -6.701111517513732e-02, -6.993161923221687e-02, -6.694589591556356e-02, -7.215438212138416e-02, -7.689527729514763e-02, -7.504348779509676e-02, -7.350039734020719e-02, -5.457705116534706e-02, -1.015348572490521e-01, -5.457705116534706e-02, -1.015348572490521e-01, -5.668022521300655e-02, -5.108287454865712e-02, -5.643666961263380e-02, -5.007250486651142e-02, -7.452621055755200e-02, -6.241995045648692e-02, -5.181874410398038e-02, -5.079502714006635e-02, -1.671904348425442e-02, -8.950848531642426e-02, -1.671904348425441e-02, -8.950848531642430e-02, -2.842918800761698e-03, -2.156282327859764e-03, -3.034041474569702e-03, -2.217737677596481e-03, -1.172726873773161e-02, -8.963126774910471e-03, -1.418919258404499e-03, -1.531191243554073e-03, -3.693977212911456e-04, -4.197687117435120e-03, -3.693977212911458e-04, -4.197687117435120e-03, -6.602160387464148e-02, -6.605326354490965e-02, -6.597929369040313e-02, -6.601674437497949e-02, -6.601810294014968e-02, -6.605351319308291e-02, -6.598508197235275e-02, -6.601692832040358e-02, -6.599899771324531e-02, -6.603568874184962e-02, -6.599899771324531e-02, -6.603568874184962e-02, -8.369617569343914e-02, -8.371151952153993e-02, -8.307851712968775e-02, -8.317629274815666e-02, -8.600108299528272e-02, -8.416824079110800e-02, -8.544783525272311e-02, -8.365099762935291e-02, -8.082255798084102e-02, -8.470269355031067e-02, -8.082255798084102e-02, -8.470269355031067e-02, -5.744982906803901e-02, -5.773786845379027e-02, -5.205621295172663e-02, -5.186902898773896e-02, -6.092288535948754e-02, -5.473439422011162e-02, -5.326614438324620e-02, -4.953409704281700e-02, -5.335790792405438e-02, -6.047445727505943e-02, -5.335790792405438e-02, -6.047445727505943e-02, -3.538057134329453e-02, -3.469414757651510e-02, -6.901598915180326e-02, -6.838348800685498e-02, -3.119605532989865e-02, -2.879661436568303e-02, -6.025897541361700e-02, -6.029054622052941e-02, -4.434299627231175e-02, -3.729260459994922e-02, -4.434299627231175e-02, -3.729260459994922e-02, -1.248687430073331e-03, -1.049138639597635e-03, -1.492136170612262e-03, -1.393951392604181e-03, -1.260273804374074e-03, -9.723802317228131e-04, -1.441220915171826e-02, -1.415143191421245e-02, -2.356801006424956e-03, -5.747794195799963e-04, -2.356801006424956e-03, -5.747794195799962e-04, -4.965247169958154e-02, -4.916789358022219e-02, -5.110450254886959e-02, -5.056941127907511e-02, -5.059477689942758e-02, -5.007023008644361e-02, -5.016467575011591e-02, -4.966917888297928e-02, -5.037920807223428e-02, -4.986963798033080e-02, -5.037920807223428e-02, -4.986963798033080e-02, -4.883104091919791e-02, -4.844781108501073e-02, -6.316336582655678e-02, -6.223252243398927e-02, -5.821103307630521e-02, -5.731516854340900e-02, -5.380577660680617e-02, -5.328451016020880e-02, -5.584106047658932e-02, -5.526393807964083e-02, -5.584106047658932e-02, -5.526393807964083e-02, -5.295287960781035e-02, -5.255070710317587e-02, -7.354183786805575e-02, -7.247679622241877e-02, -6.994066454885524e-02, -6.757723275272030e-02, -5.784049649917875e-02, -5.690438374355986e-02, -6.257270750480845e-02, -6.236705849871604e-02, -6.257270750480846e-02, -6.236705849871600e-02, -6.228576214189575e-02, -6.074332007283671e-02, -1.026560063883465e-02, -9.927061713815631e-03, -1.221373335869461e-02, -1.045453231953229e-02, -5.316275642349530e-02, -5.067244225565976e-02, -2.307911852867985e-02, -1.942415860079160e-02, -2.307911852867983e-02, -1.942415860079162e-02, -3.712704968890614e-03, -3.154504684398392e-03, -3.976669314362071e-04, -3.935948269252801e-04, -9.395596028359638e-04, -7.113234345648327e-04, -1.943093011642895e-02, -1.861692539039415e-02, -2.088356901934606e-03, -5.873302767385831e-04, -2.088356901934604e-03, -5.873302767385827e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_tm_lyp_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.147945322057634e-11, 1.847301675427200e-11, 4.148091617049851e-11, 4.147837198375103e-11, 1.847275369335338e-11, 4.148083886452218e-11, 4.147821503595200e-11, 1.847150303658708e-11, 4.147548789363683e-11, 4.148114646241593e-11, 1.847541885400883e-11, 4.148980142776584e-11, 4.147374116746520e-11, 1.847337868861712e-11, 4.148822116252364e-11, 4.147374116746520e-11, 1.847337868861712e-11, 4.148822116252364e-11, 3.729275841580359e-07, 2.392958189976293e-07, 3.730491006283623e-07, 3.729149203277743e-07, 2.392930618991588e-07, 3.730536779419213e-07, 3.729366089359796e-07, 2.391980348877677e-07, 3.727533263207664e-07, 3.727940534532757e-07, 2.391207134662467e-07, 3.726691457680672e-07, 3.746046013992662e-07, 2.392635508183138e-07, 3.712732747666262e-07, 3.746046013992662e-07, 2.392635508183138e-07, 3.712732747666262e-07, 1.369982722362383e-03, 1.537352219801414e-03, 1.292582673050316e-03, 1.386202466777708e-03, 1.546325658587755e-03, 1.290682965969553e-03, 1.447374718761844e-03, 1.776637624142536e-03, 1.600110099529444e-03, 1.487050539751997e-03, 1.701506635664321e-03, 1.440917494930112e-03, 8.162586354976779e-04, 1.842340436740333e-03, 2.214638875269175e-03, 8.162586354976779e-04, 1.842340436740333e-03, 2.214638875269175e-03, 4.996049821451415e-01, 7.244553349187997e-01, 3.916564441732257e-01, 4.825044848513583e-01, 6.876243420781538e-01, 3.646121679527192e-01, 8.015533756432589e-04, 7.080582943884520e-04, 4.726835907157422e-04, 1.085895434943385e+00, 1.772022233780902e+00, 1.012015321976249e+00, 3.324042126359277e-02, 2.204210732999284e+00, 1.880889012429507e+00, 3.324042126359314e-02, 2.204210732999285e+00, 1.880889012429507e+00, 2.677066315314597e-16, 5.310294575783324e-16, 2.686758042746382e-16, 3.080334997883765e-15, 6.104958724415614e-15, 3.088837019677512e-15, 1.108530132840785e+00, 2.021828310485412e+00, 1.023406393950597e+00, 2.719425489012619e-32, 5.396629395053357e-32, 2.709661718286472e-32, 9.791832842789421e-25, 1.991581538795577e-24, 9.886555243065320e-25, 9.791832842789421e-25, 1.991581538795578e-24, 9.886555243065324e-25, 4.899001525249008e-08, 2.802355891871077e-08, 4.928611082455310e-08, 4.894098086165669e-08, 2.799638518012080e-08, 4.924461389976358e-08, 4.898501234152784e-08, 2.802220844955965e-08, 4.928660483080046e-08, 4.894743728943644e-08, 2.799821214110047e-08, 4.924425645626576e-08, 4.896451142702253e-08, 2.800983688640180e-08, 4.926589647366383e-08, 4.896451142702253e-08, 2.800983688640180e-08, 4.926589647366383e-08, 6.676720952016612e-06, 5.225722871315213e-06, 6.683931662523416e-06, 6.423947546125207e-06, 5.026532726758676e-06, 6.461469144499109e-06, 6.987109804244809e-06, 5.377714107626647e-06, 6.734195938378940e-06, 6.771106595264210e-06, 5.195176427181188e-06, 6.515592154532084e-06, 6.162421809465440e-06, 5.048787143469867e-06, 6.769421713918289e-06, 6.162421809465440e-06, 5.048787143469867e-06, 6.769421713918289e-06, 2.480754189157853e-03, 3.047796122826569e-03, 2.573041423528486e-03, 1.706552396130942e-03, 1.997345235221597e-03, 1.695475607059934e-03, 4.548095240414269e-03, 4.596333849584244e-03, 2.805263162606481e-03, 3.727752647147963e-03, 3.770218281370987e-03, 2.387384185911925e-03, 1.548706209041712e-03, 2.924777531569530e-03, 3.205567422098598e-03, 1.548706209041715e-03, 2.924777531569530e-03, 3.205567422098598e-03, 2.425975223973159e+00, 4.205882303829982e+00, 2.340414709071299e+00, 4.493406749869807e-01, 7.163469033714480e-01, 4.352916287168915e-01, 2.924465028402076e+00, 4.777326206137007e+00, 2.430991010395714e+00, 7.323767722189668e-06, 5.797514895773395e-06, 7.394074342337086e-06, 1.971369724345267e+00, 3.053536524663118e+00, 1.534947384963593e+00, 1.971369724345267e+00, 3.053536524663118e+00, 1.534947384963593e+00, 8.821741015290985e-44, 1.765137023975679e-43, 8.911079224503010e-44, 3.542916428898866e-33, 7.055983342897957e-33, 3.554580748315061e-33, 2.061608527042286e-45, 4.136646768316621e-45, 2.093303497989439e-45, 3.213844089646988e+00, 5.869325086681656e+00, 3.166192312309497e+00, 1.157602196654942e-36, 2.381239046584156e-36, 1.227047977754359e-36, 1.157602196654941e-36, 2.381239046584156e-36, 1.227047977754359e-36, 2.535201627055056e-03, 2.974100611909241e-03, 2.403634264381600e-03, 2.606065886813453e-03, 3.062281958186818e-03, 2.469877463765125e-03, 2.581079805750635e-03, 3.030911220076296e-03, 2.446110361894372e-03, 2.560117257700312e-03, 3.005342850980560e-03, 2.427326632179085e-03, 2.570577775178865e-03, 3.018109016732376e-03, 2.436713910539359e-03, 2.570577775178865e-03, 3.018109016732376e-03, 2.436713910539359e-03, 2.877265410176509e-03, 3.418688844929681e-03, 2.750770302590394e-03, 5.929985756697746e-03, 7.350386157553491e-03, 5.616095777500819e-03, 4.786539895169845e-03, 5.843818215564887e-03, 4.520898036293890e-03, 3.876717980665358e-03, 4.694498451769143e-03, 3.701296866086482e-03, 4.305842789927824e-03, 5.244453050470894e-03, 4.102590776425100e-03, 4.305842789927824e-03, 5.244453050470894e-03, 4.102590776425100e-03, 1.371764570577753e-03, 1.567139594028025e-03, 1.339381403832094e-03, 1.933100411682160e-01, 2.937904730708272e-01, 1.850405105114732e-01, 9.548165491370338e-02, 1.369927373936727e-01, 8.763138120788402e-02, 2.883471109610587e-02, 3.929502138765541e-02, 2.724460938000784e-02, 5.123216796865385e-02, 7.439326660979330e-02, 5.140270060594821e-02, 5.123216796865388e-02, 7.439326660979330e-02, 5.140270060594822e-02, 8.821501396791463e-03, 1.105327736990439e-02, 8.122789682038715e-03, 8.853362848652979e-01, 1.672686631964185e+00, 8.769724140607584e-01, 2.493065844333171e+00, 4.511760493366183e+00, 2.337604433121376e+00, 3.542134445672300e-02, 4.664215361838203e-02, 3.046885246280857e-02, 3.394851381899803e+00, 5.671308562241651e+00, 2.873127249524956e+00, 3.394851381899804e+00, 5.671308562241651e+00, 2.873127249524955e+00, 3.386128209320866e-10, 6.666426141237915e-10, 3.376049135443861e-10, 2.035082137617844e-144, 4.065434970478615e-144, 2.036810332067668e-144, 1.713974313686758e-64, 3.449672662200811e-64, 1.746862459484112e-64, 3.318244965075048e+00, 5.883801593040785e+00, 3.166108494306175e+00, 3.991476348098458e-40, 8.210113079972192e-40, 4.235798048501067e-40, 3.991476348098138e-40, 8.210113079971531e-40, 4.235798048500727e-40]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_golden_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_golden", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.344907952599002e+03, 2.344912820620193e+03, 2.344951672343325e+03, 2.344878893660577e+03, 2.344914580742113e+03, 2.344914580742113e+03, 6.496413138711236e+01, 6.496234128015803e+01, 6.492416516821820e+01, 6.501525365226372e+01, 6.496512810625927e+01, 6.496512810625927e+01, 2.673842731424630e+00, 2.675847998383098e+00, 2.748962428863973e+00, 2.783076701023788e+00, 2.801263864510138e+00, 2.801263864510138e+00, 3.134188159019237e-01, 3.108913758720463e-01, 3.546641792361803e+00, 2.997335216688956e-01, 3.245962746129493e-01, 3.245962746129493e-01, 2.109407193651089e-01, 2.052089212496712e-01, 2.708361127844592e-01, 2.213754531501263e-01, 1.983783371964124e-01, 1.983783371964122e-01, 1.348479807304592e+02, 1.348017469268833e+02, 1.348464903771836e+02, 1.348056634828500e+02, 1.348241832093640e+02, 1.348241832093640e+02, 2.522209975428796e+01, 2.540594523444768e+01, 2.537598104792615e+01, 2.553641251615928e+01, 2.525240736854619e+01, 2.525240736854619e+01, 1.809017916247070e+00, 1.920848232364497e+00, 1.581106612171955e+00, 1.530832441389847e+00, 1.845799623975757e+00, 1.845799623975758e+00, 3.041232950521487e-01, 4.245100805928624e-01, 2.923471744299159e-01, 1.738169438912423e+01, 2.669029025752489e-01, 2.669029025752489e-01, 1.929489888706664e-01, 2.041621033520380e-01, 8.143077795387804e-02, 2.488510856727192e-01, 1.195893496452247e-01, 1.195893496452248e-01, 1.598884588127123e+00, 1.621456316317387e+00, 1.613528587761593e+00, 1.606932785149948e+00, 1.610227668577631e+00, 1.610227668577631e+00, 1.501864069058434e+00, 1.430709589266113e+00, 1.442274000300151e+00, 1.458280425109637e+00, 1.448000982338559e+00, 1.448000982338559e+00, 2.119712388516927e+00, 5.252707031061457e-01, 5.982237391856726e-01, 7.414073012273834e-01, 6.527352532572734e-01, 6.527352532572734e-01, 1.220431993865691e+00, 2.840500511569221e-01, 2.816704549035013e-01, 6.373785908850236e-01, 2.305654198345696e-01, 2.305654198345696e-01, 2.349168108268261e-01, 9.261988065255963e-02, 1.324243667576709e-01, 2.305406395911610e-01, 1.072347463278660e-01, 1.072347463278659e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_golden_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_golden", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.170448609642459e+03, 3.170441203491565e+03, 3.170478610846037e+03, 3.170462992765165e+03, 3.170565908704928e+03, 3.170590515084307e+03, 3.170227318564031e+03, 3.170158882726760e+03, 3.170466941419603e+03, 3.170325030809779e+03, 3.170466941419603e+03, 3.170325030809779e+03, 8.122391612644887e+01, 8.123488497852922e+01, 8.122607554501270e+01, 8.123756801722794e+01, 8.128372374921989e+01, 8.129977559193358e+01, 8.120996497759377e+01, 8.122677125626484e+01, 8.119155662444004e+01, 8.128086186083007e+01, 8.119155662444004e+01, 8.128086186083007e+01, 2.736763642863300e+00, 2.793077672958922e+00, 2.716329694536424e+00, 2.785466253347297e+00, 2.442640469659932e+00, 2.343816679860594e+00, 2.428795126201688e+00, 2.461247786347152e+00, 2.886737649800108e+00, 1.888127156880179e+00, 2.886737649800108e+00, 1.888127156880179e+00, 3.791632666835108e-02, 6.051684129413742e-02, 4.995298373585531e-02, 7.572390926307956e-02, 3.635532861091492e+00, 4.060361810190185e+00, -9.150905363328719e-02, -8.989405949841488e-02, 4.681297156739540e-02, -1.585631815266006e-01, 4.681297156739526e-02, -1.585631815266005e-01, -2.099569067401565e-01, -2.109512967117424e-01, -2.044763102199788e-01, -2.049168439876299e-01, -2.574810501407815e-01, -2.546744493094863e-01, -2.197404737889696e-01, -2.227941431440409e-01, -2.102414626894157e-01, -1.321159576655591e-01, -2.102414626894153e-01, -1.321159576655590e-01, 1.929144543589710e+02, 1.928180889124222e+02, 1.930839132239635e+02, 1.929817755685964e+02, 1.929238266290881e+02, 1.928236831138098e+02, 1.930692363682708e+02, 1.929723879694120e+02, 1.930013126998729e+02, 1.929003303298537e+02, 1.930013126998729e+02, 1.929003303298537e+02, 2.108064032696026e+01, 2.107801108733616e+01, 2.157766217388141e+01, 2.156156161344444e+01, 2.032690385631527e+01, 2.054562722041207e+01, 2.075814323037842e+01, 2.097824115378677e+01, 2.190846081579832e+01, 2.140931007271151e+01, 2.190846081579832e+01, 2.140931007271151e+01, 2.358476305687967e+00, 2.346711963329904e+00, 2.987984387626357e+00, 2.993032261145528e+00, 1.900699347609438e+00, 2.065927112329370e+00, 2.246233055319043e+00, 2.395495734789859e+00, 2.585681583028619e+00, 2.324432677642379e+00, 2.585681583028619e+00, 2.324432677642381e+00, -2.059453672886639e-01, -2.008624726955743e-01, -6.185012925445282e-02, -5.971341686417776e-02, -2.093999905583472e-01, -2.027794702181409e-01, 2.717754004463064e+01, 2.715421812844322e+01, -1.529642050119518e-01, -1.062385871686272e-01, -1.529642050119518e-01, -1.062385871686272e-01, -1.855161185465476e-01, -1.994059492619182e-01, -1.998602010622482e-01, -2.080003937464558e-01, -8.371593293110924e-02, -7.935540191274357e-02, -2.077223737442269e-01, -2.119289582418667e-01, -1.187526527496168e-01, -1.197295475495530e-01, -1.187526527496169e-01, -1.197295475495532e-01, 2.592856129513350e+00, 2.613631047637504e+00, 2.522100335425343e+00, 2.543101390306867e+00, 2.546873998153276e+00, 2.567950724008598e+00, 2.567587726642974e+00, 2.588293960770175e+00, 2.557227288305324e+00, 2.578111001449449e+00, 2.557227288305324e+00, 2.578111001449449e+00, 2.467321467488778e+00, 2.483753186784826e+00, 1.491506182499527e+00, 1.508961366503701e+00, 1.761601716359956e+00, 1.781662648203888e+00, 2.041330184458035e+00, 2.056990669986344e+00, 1.900083239105578e+00, 1.915765046586273e+00, 1.900083239105578e+00, 1.915765046586273e+00, 3.270921002093301e+00, 3.285569086593502e+00, 5.287941405645460e-02, 5.779600717069413e-02, 2.318188758441014e-01, 2.465090524987290e-01, 7.145171987115081e-01, 7.246522707966211e-01, 4.513125942826892e-01, 4.530194716428318e-01, 4.513125942826891e-01, 4.530194716428316e-01, 1.251419362469175e+00, 1.276146758595444e+00, -2.706782733014353e-01, -2.702749956984801e-01, -2.591137531974120e-01, -2.539512644037075e-01, 7.069135770243393e-01, 7.339304129605886e-01, -1.821688927852853e-01, -1.542876609961883e-01, -1.821688927852853e-01, -1.542876609961883e-01, -2.353612734950044e-01, -2.328529390381547e-01, -1.058109676034912e-01, -7.949888026259026e-02, -1.346495318854082e-01, -1.304937498829279e-01, -1.745862610016331e-01, -1.749215652507974e-01, -8.268393157446784e-02, -1.175378891665729e-01, -8.268393157446766e-02, -1.175378891665728e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_golden_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_golden", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.735648753927643e-06, 0.000000000000000e+00, 3.735663185261364e-06, 3.735612547613978e-06, 0.000000000000000e+00, 3.735636882401917e-06, 3.735489574473961e-06, 0.000000000000000e+00, 3.735462671604603e-06, 3.735898654610772e-06, 0.000000000000000e+00, 3.735984027594866e-06, 3.735627520489190e-06, 0.000000000000000e+00, 3.735770357952453e-06, 3.735627520489190e-06, 0.000000000000000e+00, 3.735770357952453e-06, 8.656168512953502e-04, 0.000000000000000e+00, 8.656522838637181e-04, 8.656087325449925e-04, 0.000000000000000e+00, 8.656491925623441e-04, 8.654625295445959e-04, 0.000000000000000e+00, 8.654090791473112e-04, 8.652968352913244e-04, 0.000000000000000e+00, 8.652604045503625e-04, 8.660512316091200e-04, 0.000000000000000e+00, 8.650798126959542e-04, 8.660512316091200e-04, 0.000000000000000e+00, 8.650798126959542e-04, 1.213539897757505e-01, 0.000000000000000e+00, 1.199222738044546e-01, 1.219223485385283e-01, 0.000000000000000e+00, 1.201580544609286e-01, 1.298669422674484e-01, 0.000000000000000e+00, 1.325882789378107e-01, 1.284164010959728e-01, 0.000000000000000e+00, 1.275854175317843e-01, 1.172048774777548e-01, 0.000000000000000e+00, 1.426316745657260e-01, 1.172048774777548e-01, 0.000000000000000e+00, 1.426316745657260e-01, 7.432160365090923e+00, 0.000000000000000e+00, 6.628198369288989e+00, 7.164053558010561e+00, 0.000000000000000e+00, 6.289283838019381e+00, 8.005540628016009e-02, 0.000000000000000e+00, 7.249562825511502e-02, 1.635044133891114e+01, 0.000000000000000e+00, 1.569177210303693e+01, 6.528166501077521e+00, 0.000000000000000e+00, 4.227913853554925e+01, 6.528166501077523e+00, 0.000000000000000e+00, 4.227913853554924e+01, 1.827078648556544e+05, 0.000000000000000e+00, 1.522739034235665e+05, 1.587895899032734e+05, 0.000000000000000e+00, 1.291655945923331e+05, 9.236239771851796e+02, 0.000000000000000e+00, 7.843057159902534e+02, 8.411173890345508e+05, 0.000000000000000e+00, 8.846049961540429e+05, 2.573493343881710e+05, 0.000000000000000e+00, 1.396503665907533e+06, 2.573493343881710e+05, 0.000000000000000e+00, 1.396503665907532e+06, 2.578266525574029e-04, 0.000000000000000e+00, 2.580120195659614e-04, 2.576822617938167e-04, 0.000000000000000e+00, 2.578724143106882e-04, 2.578178567026150e-04, 0.000000000000000e+00, 2.580066682928328e-04, 2.576939687403821e-04, 0.000000000000000e+00, 2.578798496148262e-04, 2.577532771846867e-04, 0.000000000000000e+00, 2.579419890633350e-04, 2.577532771846867e-04, 0.000000000000000e+00, 2.579419890633350e-04, 4.856264894708488e-03, 0.000000000000000e+00, 4.857001150050238e-03, 4.750525583917457e-03, 0.000000000000000e+00, 4.754406134433636e-03, 4.947434551037323e-03, 0.000000000000000e+00, 4.921851978970583e-03, 4.853532975872584e-03, 0.000000000000000e+00, 4.827391701007722e-03, 4.731009733337881e-03, 0.000000000000000e+00, 4.793714561936551e-03, 4.731009733337881e-03, 0.000000000000000e+00, 4.793714561936551e-03, 1.795200481801964e-01, 0.000000000000000e+00, 1.809534685031026e-01, 1.406761085554489e-01, 0.000000000000000e+00, 1.404846899368192e-01, 2.405149353377682e-01, 0.000000000000000e+00, 2.158254538201093e-01, 2.129677640484036e-01, 0.000000000000000e+00, 1.930754854391888e-01, 1.595776974079128e-01, 0.000000000000000e+00, 1.860271038098130e-01, 1.595776974079128e-01, 0.000000000000000e+00, 1.860271038098130e-01, 4.854265006096538e+01, 0.000000000000000e+00, 4.717999427682498e+01, 7.027844649308959e+00, 0.000000000000000e+00, 6.923667383713144e+00, 6.557308107179986e+01, 0.000000000000000e+00, 5.604500956558464e+01, 5.142782524521276e-03, 0.000000000000000e+00, 5.149720209761438e-03, 3.240277553577437e+01, 0.000000000000000e+00, 2.725719286336919e+01, 3.240277553577437e+01, 0.000000000000000e+00, 2.725719286336919e+01, 1.992248886960423e+06, 0.000000000000000e+00, 1.774978809129800e+06, 9.442626585431938e+05, 0.000000000000000e+00, 9.026541020885301e+05, 2.258307957778603e+06, 0.000000000000000e+00, 1.901305279217765e+06, 2.004797722350623e+02, 0.000000000000000e+00, 1.968267187600706e+02, 2.119639573850650e+06, 0.000000000000000e+00, 8.183639501196553e+05, 2.119639573850649e+06, 0.000000000000000e+00, 8.183639501196558e+05, 1.786678296726882e-01, 0.000000000000000e+00, 1.766117348969592e-01, 1.817887337300781e-01, 0.000000000000000e+00, 1.796758059632307e-01, 1.806856694305441e-01, 0.000000000000000e+00, 1.785862967160333e-01, 1.797744672534002e-01, 0.000000000000000e+00, 1.777046385066549e-01, 1.802296891073712e-01, 0.000000000000000e+00, 1.781453254863023e-01, 1.802296891073712e-01, 0.000000000000000e+00, 1.781453254863023e-01, 1.938586323189871e-01, 0.000000000000000e+00, 1.919487567667727e-01, 3.069330202156509e-01, 0.000000000000000e+00, 3.029781447868730e-01, 2.673460446533207e-01, 0.000000000000000e+00, 2.638179812918612e-01, 2.341727413430562e-01, 0.000000000000000e+00, 2.317197430963809e-01, 2.502841298092531e-01, 0.000000000000000e+00, 2.475155271990306e-01, 2.502841298092531e-01, 0.000000000000000e+00, 2.475155271990306e-01, 1.223051129574026e-01, 0.000000000000000e+00, 1.217091604259506e-01, 3.488850966919570e+00, 0.000000000000000e+00, 3.428943615287531e+00, 2.027132902824883e+00, 0.000000000000000e+00, 1.967343163740112e+00, 8.737114640583551e-01, 0.000000000000000e+00, 8.593773883510123e-01, 1.316776873558004e+00, 0.000000000000000e+00, 1.318171400248086e+00, 1.316776873558005e+00, 0.000000000000000e+00, 1.318171400248086e+00, 3.948585568523273e-01, 0.000000000000000e+00, 3.868124980059965e-01, 9.774550834687714e+02, 0.000000000000000e+00, 9.576625651689460e+02, 4.016054651142775e+02, 0.000000000000000e+00, 3.638311291865656e+02, 9.893936527835474e-01, 0.000000000000000e+00, 9.459521297059298e-01, 1.047426571742675e+02, 0.000000000000000e+00, 8.897349299528332e+01, 1.047426571742676e+02, 0.000000000000000e+00, 8.897349299528335e+01, 6.242318724774940e+04, 0.000000000000000e+00, 5.609792128462261e+04, 4.849070614393976e+07, 0.000000000000000e+00, 4.815973373086896e+07, 5.755902786365206e+06, 0.000000000000000e+00, 4.784120143295009e+06, 1.190934886595293e+02, 0.000000000000000e+00, 1.134367184073331e+02, 2.494134803063256e+06, 0.000000000000000e+00, 1.062782274847179e+06, 2.494134803063261e+06, 0.000000000000000e+00, 1.062782274847183e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lv_rpw86_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lv_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.034092609347767e+01, -2.034096746640698e+01, -2.034119354228710e+01, -2.034057713262373e+01, -2.034089412647160e+01, -2.034089412647160e+01, -3.343903639216153e+00, -3.343887121565404e+00, -3.343587493348598e+00, -3.344743855191042e+00, -3.343964703064062e+00, -3.343964703064062e+00, -6.602248480256020e-01, -6.599675005441621e-01, -6.554440525614798e-01, -6.601655567232767e-01, -6.618889559316766e-01, -6.618889559316766e-01, -2.019134491374401e-01, -2.021707168249372e-01, -7.655736189941991e-01, -1.839472248916615e-01, -1.999684854407605e-01, -1.999684854407604e-01, -2.408411512396919e-02, -2.471434099249197e-02, -7.587144091950734e-02, -1.744755124084792e-02, -2.081896653773282e-02, -2.081896653773281e-02, -4.923869081456079e+00, -4.924037463963549e+00, -4.923883708439543e+00, -4.924032235042333e+00, -4.923950571614174e+00, -4.923950571614174e+00, -1.976275809361896e+00, -1.986913933367501e+00, -1.974394219914347e+00, -1.983715948091624e+00, -1.983177334785748e+00, -1.983177334785748e+00, -5.614459216585683e-01, -5.964930546779136e-01, -5.220412133503939e-01, -5.300495705061106e-01, -5.693710263757708e-01, -5.693710263757708e-01, -1.483437569650811e-01, -2.267040914369482e-01, -1.399032494768789e-01, -1.795599810216949e+00, -1.607273116450748e-01, -1.607273116450748e-01, -1.452943308226839e-02, -1.693563810999461e-02, -1.200905725281436e-02, -1.037297700931755e-01, -1.484352149274876e-02, -1.484352149274876e-02, -5.487153074853650e-01, -5.483630790282730e-01, -5.484853828433431e-01, -5.485837381141104e-01, -5.485338068883443e-01, -5.485338068883443e-01, -5.330357332890832e-01, -4.838415113124813e-01, -4.968922239786111e-01, -5.105117892724187e-01, -5.033606288510653e-01, -5.033606288510653e-01, -6.258580609016021e-01, -2.598100381330423e-01, -2.873416591689322e-01, -3.448327686685020e-01, -3.119797017470184e-01, -3.119797017470184e-01, -4.461306708343272e-01, -7.423694356867441e-02, -9.120945683986342e-02, -3.263151184619392e-01, -1.211556339640458e-01, -1.211556339640458e-01, -3.029524464416446e-02, -6.536093215206712e-03, -1.098566523709939e-02, -1.159564109444976e-01, -1.380779964607389e-02, -1.380779964607388e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lv_rpw86_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lv_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.574166467657607e+01, -2.574163340028623e+01, -2.574177130105354e+01, -2.574171084569968e+01, -2.574209745177664e+01, -2.574218238436052e+01, -2.574089368665508e+01, -2.574064795769770e+01, -2.574172902518086e+01, -2.574125031204101e+01, -2.574172902518086e+01, -2.574125031204101e+01, -4.148454308854858e+00, -4.148596613845961e+00, -4.148492192393096e+00, -4.148639109864412e+00, -4.149457398052492e+00, -4.149734475106387e+00, -4.148453190863087e+00, -4.148730517220755e+00, -4.147682196466743e+00, -4.149664501999276e+00, -4.147682196466743e+00, -4.149664501999276e+00, -7.771637247536578e-01, -7.829507776866792e-01, -7.750149692450018e-01, -7.821285894708723e-01, -7.458083695043950e-01, -7.353249734784457e-01, -7.459996719050461e-01, -7.493862754850510e-01, -7.928849256843495e-01, -6.883334849394105e-01, -7.928849256843495e-01, -6.883334849394105e-01, -1.511152907667301e-01, -1.630851262639990e-01, -1.564137888809096e-01, -1.703029992914584e-01, -8.944070611139662e-01, -9.361203301028355e-01, -1.402482863882703e-01, -1.401806011327433e-01, -1.593874361212011e-01, -1.414534829857885e-01, -1.593874361212010e-01, -1.414534829857882e-01, -1.902389738060674e-02, -1.977385454800421e-02, -1.948769498954202e-02, -2.035205870025050e-02, -6.624938690298920e-02, -6.904715233342718e-02, -1.404813387101538e-02, -1.394322952926097e-02, -1.773100274473838e-02, -1.147342678249678e-02, -1.773100274473841e-02, -1.147342678249674e-02, -6.320982663502339e+00, -6.319428348732994e+00, -6.323152472337121e+00, -6.321524647774405e+00, -6.321105185538720e+00, -6.319501792553415e+00, -6.322967027356012e+00, -6.321406196251433e+00, -6.322092866480620e+00, -6.320481257315909e+00, -6.322092866480620e+00, -6.320481257315909e+00, -2.209521726317414e+00, -2.209396467833896e+00, -2.230811308492511e+00, -2.230086319814414e+00, -2.182282751536772e+00, -2.190134486548625e+00, -2.200914125595481e+00, -2.208840774049295e+00, -2.241200612560648e+00, -2.223045585727819e+00, -2.241200612560648e+00, -2.223045585727819e+00, -7.044430492704958e-01, -7.026414549153563e-01, -7.817849427343639e-01, -7.823350744801361e-01, -6.350689122656455e-01, -6.605915019542866e-01, -6.789322252043413e-01, -7.012601330835860e-01, -7.356208383012456e-01, -6.980586609780176e-01, -7.356208383012456e-01, -6.980586609780176e-01, -1.420332037842373e-01, -1.426197066584712e-01, -1.516961491923209e-01, -1.517882390030051e-01, -1.316251182295408e-01, -1.366604037529845e-01, -2.356981444297079e+00, -2.355952891291476e+00, -1.484534981597672e-01, -1.386413999708462e-01, -1.484534981597672e-01, -1.386413999708462e-01, -1.141485347284458e-02, -1.185159946400567e-02, -1.346988063164843e-02, -1.370027599600412e-02, -9.519072451318774e-03, -9.756523243217503e-03, -9.729445729402740e-02, -9.796441950973196e-02, -1.032483085345703e-02, -1.255044759515613e-02, -1.032483085345699e-02, -1.255044759515613e-02, -7.259934578612434e-01, -7.288623485099559e-01, -7.180729569594222e-01, -7.209917985759777e-01, -7.208527584018620e-01, -7.237712698732957e-01, -7.231704223076774e-01, -7.260404841323286e-01, -7.220116078767328e-01, -7.249050737634966e-01, -7.220116078767328e-01, -7.249050737634966e-01, -7.076095393622095e-01, -7.099572766647879e-01, -5.722496131346789e-01, -5.752086276466615e-01, -6.120816644287566e-01, -6.152435798841176e-01, -6.511657677375914e-01, -6.535807011575292e-01, -6.316051725765622e-01, -6.341047012209287e-01, -6.316051725765622e-01, -6.341047012209287e-01, -8.183791142448954e-01, -8.200297575750470e-01, -1.922546817901829e-01, -1.942899724652831e-01, -2.646953795502895e-01, -2.695753164158051e-01, -3.995143958659178e-01, -4.020584240860662e-01, -3.321358977676453e-01, -3.323823361466682e-01, -3.321358977676453e-01, -3.323823361466682e-01, -5.250539540535391e-01, -5.295291910826316e-01, -6.564983865988089e-02, -6.598678601922764e-02, -8.250935834043925e-02, -8.453311483604760e-02, -3.910813466566482e-01, -3.978385557522449e-01, -1.145674628678638e-01, -1.176041090096896e-01, -1.145674628678640e-01, -1.176041090096896e-01, -2.432843068220576e-02, -2.483835380758187e-02, -5.377311465913896e-03, -5.086325624960118e-03, -8.652246351769553e-03, -8.924687446682664e-03, -1.102247782049807e-01, -1.116983288364253e-01, -9.306253470290128e-03, -1.185560882172840e-02, -9.306253470290096e-03, -1.185560882172841e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lv_rpw86_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lv_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.985766352499416e-09, 0.000000000000000e+00, -6.985802442339485e-09, -6.985677420180540e-09, 0.000000000000000e+00, -6.985737834637284e-09, -6.985373306074410e-09, 0.000000000000000e+00, -6.985307552818969e-09, -6.986378173086362e-09, 0.000000000000000e+00, -6.986588201416462e-09, -6.985714301605796e-09, 0.000000000000000e+00, -6.986061824817532e-09, -6.985714301605796e-09, 0.000000000000000e+00, -6.986061824817532e-09, -9.922636035621366e-06, 0.000000000000000e+00, -9.923261527956307e-06, -9.922522454792849e-06, 0.000000000000000e+00, -9.923230138185645e-06, -9.920597065488454e-06, 0.000000000000000e+00, -9.919861336831626e-06, -9.917529519979058e-06, 0.000000000000000e+00, -9.917064913139700e-06, -9.929242614392133e-06, 0.000000000000000e+00, -9.914577131862918e-06, -9.929242614392133e-06, 0.000000000000000e+00, -9.914577131862918e-06, -7.204570405137532e-03, 0.000000000000000e+00, -7.092863116928204e-03, -7.249181117272329e-03, 0.000000000000000e+00, -7.111283859443399e-03, -7.883969251532780e-03, 0.000000000000000e+00, -8.106469767763053e-03, -7.767316117713038e-03, 0.000000000000000e+00, -7.699980699222040e-03, -6.881966993280982e-03, 0.000000000000000e+00, -8.962554579081681e-03, -6.881966993280982e-03, 0.000000000000000e+00, -8.962554579081681e-03, -1.793376093968309e+00, 0.000000000000000e+00, -1.539109969527197e+00, -1.709002169749140e+00, 0.000000000000000e+00, -1.429994731316441e+00, -4.137771484663379e-03, 0.000000000000000e+00, -3.629064651885800e-03, -2.888373676304627e+00, 0.000000000000000e+00, -2.812515916026631e+00, -1.509741731607291e+00, 0.000000000000000e+00, -4.030208105554959e+00, -1.509741731607288e+00, 0.000000000000000e+00, -4.030208105554975e+00, -4.050957366120322e+03, 0.000000000000000e+00, -3.483821226291714e+03, -3.693415533082911e+03, 0.000000000000000e+00, -3.120426824685306e+03, -4.290616043373870e+01, 0.000000000000000e+00, -3.754465235049781e+01, -1.334195293494314e+04, 0.000000000000000e+00, -1.374085101927922e+04, -5.333702104441842e+03, 0.000000000000000e+00, -3.004905603304893e+04, -5.333702104441837e+03, 0.000000000000000e+00, -3.004905603304904e+04, -1.980778107525425e-06, 0.000000000000000e+00, -1.982674439356648e-06, -1.979365343917156e-06, 0.000000000000000e+00, -1.981308204973668e-06, -1.980691399426916e-06, 0.000000000000000e+00, -1.982621606161988e-06, -1.979479251886616e-06, 0.000000000000000e+00, -1.981380514562760e-06, -1.980060686217093e-06, 0.000000000000000e+00, -1.981989181373343e-06, -1.980060686217093e-06, 0.000000000000000e+00, -1.981989181373343e-06, -9.861163203983268e-05, 0.000000000000000e+00, -9.863159012957366e-05, -9.575137688332754e-05, 0.000000000000000e+00, -9.585585026695599e-05, -1.011163755779050e-04, 0.000000000000000e+00, -1.004104266542927e-04, -9.855518899682841e-05, 0.000000000000000e+00, -9.783999523608008e-05, -9.521801487865768e-05, 0.000000000000000e+00, -9.691537934245848e-05, -9.521801487865768e-05, 0.000000000000000e+00, -9.691537934245848e-05, -1.219247416198143e-02, 0.000000000000000e+00, -1.232267037635776e-02, -8.849040248584629e-03, 0.000000000000000e+00, -8.833307518819639e-03, -1.798741797063472e-02, 0.000000000000000e+00, -1.557778601499244e-02, -1.537525116958251e-02, 0.000000000000000e+00, -1.349008292554837e-02, -1.042945802016148e-02, 0.000000000000000e+00, -1.279236954113389e-02, -1.042945802016148e-02, 0.000000000000000e+00, -1.279236954113390e-02, -3.750360670528112e+00, 0.000000000000000e+00, -3.722738640492438e+00, -1.374046442946710e+00, 0.000000000000000e+00, -1.359357677013942e+00, -4.832371660104020e+00, 0.000000000000000e+00, -4.298752492406798e+00, -1.073751416925953e-04, 0.000000000000000e+00, -1.075685107828546e-04, -3.343990457234649e+00, 0.000000000000000e+00, -3.964206076803653e+00, -3.343990457234649e+00, 0.000000000000000e+00, -3.964206076803653e+00, -3.049111460818739e+04, 0.000000000000000e+00, -2.623992465828463e+04, -1.578797529119627e+04, 0.000000000000000e+00, -1.475078704393138e+04, -6.354242805064830e+04, 0.000000000000000e+00, -5.773849730097142e+04, -1.313094410124806e+01, 0.000000000000000e+00, -1.274583869524810e+01, -4.572549069555222e+04, 0.000000000000000e+00, -2.114778772029009e+04, -4.572549069555264e+04, 0.000000000000000e+00, -2.114778772029002e+04, -1.217833539794261e-02, 0.000000000000000e+00, -1.199188981035941e-02, -1.245622619390134e-02, 0.000000000000000e+00, -1.226383544093115e-02, -1.235844879780369e-02, 0.000000000000000e+00, -1.216756479937686e-02, -1.227734134186379e-02, 0.000000000000000e+00, -1.208933965410257e-02, -1.231790566431369e-02, 0.000000000000000e+00, -1.212848053132649e-02, -1.231790566431369e-02, 0.000000000000000e+00, -1.212848053132649e-02, -1.357901634663454e-02, 0.000000000000000e+00, -1.340094218041628e-02, -2.483171936564581e-02, 0.000000000000000e+00, -2.440735491069097e-02, -2.070586451970738e-02, 0.000000000000000e+00, -2.034478408911963e-02, -1.741380675490536e-02, 0.000000000000000e+00, -1.717172592613655e-02, -1.899589861883453e-02, 0.000000000000000e+00, -1.871715947765991e-02, -1.899589861883453e-02, 0.000000000000000e+00, -1.871715947765991e-02, -7.341802179785449e-03, 0.000000000000000e+00, -7.294577104703747e-03, -6.530407265488103e-01, 0.000000000000000e+00, -6.387392387476392e-01, -3.120002723120374e-01, 0.000000000000000e+00, -2.992986777469763e-01, -1.001339145982754e-01, 0.000000000000000e+00, -9.795136599908247e-02, -1.733738829261670e-01, 0.000000000000000e+00, -1.736022337547682e-01, -1.733738829261671e-01, 0.000000000000000e+00, -1.736022337547682e-01, -3.473895661572157e-02, 0.000000000000000e+00, -3.380151742333948e-02, -4.348924563551550e+01, 0.000000000000000e+00, -4.277464415126853e+01, -2.075556162267598e+01, 0.000000000000000e+00, -1.931153936502834e+01, -1.183382319664261e-01, 0.000000000000000e+00, -1.114887538225307e-01, -8.042737447305241e+00, 0.000000000000000e+00, -7.774901292283070e+00, -8.042737447305285e+00, 0.000000000000000e+00, -7.774901292283142e+00, -1.552838536046411e+03, 0.000000000000000e+00, -1.435709206006850e+03, -6.154192504490690e+05, 0.000000000000000e+00, -7.692188657136497e+05, -9.214677409325783e+04, 0.000000000000000e+00, -8.147028037212304e+04, -9.276152605059405e+00, 0.000000000000000e+00, -8.870825871998615e+00, -6.950198538397753e+04, 0.000000000000000e+00, -2.647959112574715e+04, -6.950198538397813e+04, 0.000000000000000e+00, -2.647959112574711e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
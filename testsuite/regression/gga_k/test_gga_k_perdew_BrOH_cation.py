
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_perdew_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_perdew", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.174868061686054e+03, 2.174877028008358e+03, 2.174925711104896e+03, 2.174792129797474e+03, 2.174860869008007e+03, 2.174860869008007e+03, 5.873082847185290e+01, 5.873028403470918e+01, 5.872058467727338e+01, 5.875977037951718e+01, 5.873301736617542e+01, 5.873301736617542e+01, 2.284022996209877e+00, 2.282045630238989e+00, 2.244700021807741e+00, 2.277541026302111e+00, 2.286718061461132e+00, 2.286718061461132e+00, 2.045168817734707e-01, 2.061440162670811e-01, 3.073552182220880e+00, 1.636955861029753e-01, 2.043666658919216e-01, 2.043666658919215e-01, 8.139652114411543e-02, 7.919865201090007e-02, 1.078057989642631e-01, 8.535002927728313e-02, 7.652102498112993e-02, 7.652102498112984e-02, 1.274997840123190e+02, 1.275097017741288e+02, 1.275005911045112e+02, 1.275093403756434e+02, 1.275046150225279e+02, 1.275046150225279e+02, 2.039178236038251e+01, 2.061800722264107e+01, 2.033840787910040e+01, 2.053676470069741e+01, 2.054440153773070e+01, 2.054440153773070e+01, 1.656340203548385e+00, 1.872232893160381e+00, 1.431890971112982e+00, 1.478539221839527e+00, 1.704906883516852e+00, 1.704906883516852e+00, 1.404283592156997e-01, 2.474353272783937e-01, 1.326082376707339e-01, 1.696594323859365e+01, 1.349713334380377e-01, 1.349713334380377e-01, 7.438106388325380e-02, 7.871464912077247e-02, 3.140168953971385e-02, 1.048951043464864e-01, 4.611913126384175e-02, 4.611913126384179e-02, 1.584704602826214e+00, 1.582323612150233e+00, 1.583159883553920e+00, 1.583828944589827e+00, 1.583490815442553e+00, 1.583490815442553e+00, 1.495515892889370e+00, 1.226910507666307e+00, 1.296586016102343e+00, 1.370401909191800e+00, 1.331519936296113e+00, 1.331519936296113e+00, 2.061027331444862e+00, 3.362564891088298e-01, 4.235740374148792e-01, 6.224663493789913e-01, 5.062125102516084e-01, 5.062125102516084e-01, 1.042927131034393e+00, 1.125993300099340e-01, 1.143764912491271e-01, 5.586515764130837e-01, 1.034867753948788e-01, 1.034867753948788e-01, 9.073700640634086e-02, 3.569722933641324e-02, 5.104544804441317e-02, 1.017050191939830e-01, 4.135294692207358e-02, 4.135294692207353e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_perdew_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_perdew", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.340468237115223e+03, 3.340459926559002e+03, 3.340493488642424e+03, 3.340478268775131e+03, 3.340573629239224e+03, 3.340593284801409e+03, 3.340288473921615e+03, 3.340229824972397e+03, 3.340483330196729e+03, 3.340374634880686e+03, 3.340483330196729e+03, 3.340374634880686e+03, 8.746079470684639e+01, 8.746348961334913e+01, 8.746197807706004e+01, 8.746465733244239e+01, 8.749051970318529e+01, 8.749901855343073e+01, 8.746926314138314e+01, 8.747731685781315e+01, 8.742954422119647e+01, 8.750598047539678e+01, 8.742954422119647e+01, 8.750598047539678e+01, 3.135656363731432e+00, 3.173888971505124e+00, 3.121309205875746e+00, 3.168210858720014e+00, 2.930341485375977e+00, 2.864946297608286e+00, 2.939955269608813e+00, 2.961154064007400e+00, 3.242280551356469e+00, 2.596124593862933e+00, 3.242280551356469e+00, 2.596124593862933e+00, 1.485443732890872e-01, 1.678766094220254e-01, 1.565041960049483e-01, 1.788848400639227e-01, 4.147099644373362e+00, 4.498552523649871e+00, 4.391167245807757e-02, 4.673456111404681e-02, 1.640681847376071e-01, -1.908052285830007e-02, 1.640681847376070e-01, -1.908052285830015e-02, -8.075721278034254e-02, -8.111998218645365e-02, -7.862950413841624e-02, -7.877361110693706e-02, -9.385291518787474e-02, -9.215189392095333e-02, -8.462888640799790e-02, -8.580763665965142e-02, -8.089922055061781e-02, -5.087555273011116e-02, -8.089922055061774e-02, -5.087555273011112e-02, 2.002619996377594e+02, 2.001645457132918e+02, 2.003743410368725e+02, 2.002730485793578e+02, 2.002684755007911e+02, 2.001684419025939e+02, 2.003648696727308e+02, 2.002670106242107e+02, 2.003193696844937e+02, 2.002190195824203e+02, 2.003193696844937e+02, 2.002190195824203e+02, 2.591086456612040e+01, 2.590805376621184e+01, 2.636537727880638e+01, 2.634934955926189e+01, 2.540119414176034e+01, 2.554630965820781e+01, 2.579325050438209e+01, 2.594225253382019e+01, 2.654246766975415e+01, 2.619192083579486e+01, 2.654246766975415e+01, 2.619192083579486e+01, 2.511758330526470e+00, 2.498748755006686e+00, 3.037279381246775e+00, 3.040937432958966e+00, 2.051374559631444e+00, 2.213807048754890e+00, 2.295949961427949e+00, 2.450099581584947e+00, 2.730220124216856e+00, 2.461040117605809e+00, 2.730220124216853e+00, 2.461040117605810e+00, -4.103910588400690e-02, -3.834566607912950e-02, 1.151839596263693e-01, 1.173984752265245e-01, -4.933374209467972e-02, -4.332093725703880e-02, 2.759367273581303e+01, 2.756930554016185e+01, -8.766332886443859e-03, 1.537353533168640e-02, -8.766332886443859e-03, 1.537353533168640e-02, -7.146413544505825e-02, -7.681458128837673e-02, -7.697144500243636e-02, -8.010699047053282e-02, -3.223378334393759e-02, -3.054966935733974e-02, -6.516382183345287e-02, -6.660132514790025e-02, -4.573536161848980e-02, -4.608446087388893e-02, -4.573536161848992e-02, -4.608446087388889e-02, 2.607184418362497e+00, 2.627646400268677e+00, 2.561456589446548e+00, 2.581986988665446e+00, 2.577467232543160e+00, 2.598072974812632e+00, 2.590839136808965e+00, 2.611229475392002e+00, 2.584148332377828e+00, 2.604642188245906e+00, 2.584148332377828e+00, 2.604642188245906e+00, 2.473671944813420e+00, 2.490089017247141e+00, 1.695794071322803e+00, 1.712255076484487e+00, 1.907980060578238e+00, 1.926644785788753e+00, 2.129279547847083e+00, 2.144773337902603e+00, 2.016591154248589e+00, 2.032194229075813e+00, 2.016591154248589e+00, 2.032194229075813e+00, 3.330716455236674e+00, 3.343113960380488e+00, 2.421165086396820e-01, 2.465865193278852e-01, 4.077892146462245e-01, 4.198704346865448e-01, 8.334726806875054e-01, 8.435674953267346e-01, 5.985048477150332e-01, 5.988631674885736e-01, 5.985048477150328e-01, 5.988631674885742e-01, 1.429975739709565e+00, 1.452602209836868e+00, -9.913833046335671e-02, -9.891179892106421e-02, -9.048980980256251e-02, -8.786238068280866e-02, 7.861441080941894e-01, 8.121656643106636e-01, -4.725133120583736e-02, -3.386824998097766e-02, -4.725133120583715e-02, -3.386824998097749e-02, -9.038211645217020e-02, -8.939149657906757e-02, -4.077482069640027e-02, -3.063434610527600e-02, -5.187691522732651e-02, -5.027322813345483e-02, -4.621228129588829e-02, -4.564667410685898e-02, -3.183795735068721e-02, -4.524913578804608e-02, -3.183795735068719e-02, -4.524913578804600e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_perdew_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_perdew", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.439683312682032e-06, 0.000000000000000e+00, 1.439688874357494e-06, 1.439669358831988e-06, 0.000000000000000e+00, 1.439678737281583e-06, 1.439621965632805e-06, 0.000000000000000e+00, 1.439611597245120e-06, 1.439779624390965e-06, 0.000000000000000e+00, 1.439812526921131e-06, 1.439675129330023e-06, 0.000000000000000e+00, 1.439730179338752e-06, 1.439675129330023e-06, 0.000000000000000e+00, 1.439730179338752e-06, 3.336009207058827e-04, 0.000000000000000e+00, 3.336145747591039e-04, 3.335977916453809e-04, 0.000000000000000e+00, 3.336133831461044e-04, 3.335414413785638e-04, 0.000000000000000e+00, 3.335208407701751e-04, 3.334775928508773e-04, 0.000000000000000e+00, 3.334635513121492e-04, 3.337683274845135e-04, 0.000000000000000e+00, 3.333939485073937e-04, 3.337683274845135e-04, 0.000000000000000e+00, 3.333939485073937e-04, 4.676878090601343e-02, 0.000000000000000e+00, 4.621700707936227e-02, 4.698782237522905e-02, 0.000000000000000e+00, 4.630787525491357e-02, 5.004960986476509e-02, 0.000000000000000e+00, 5.109839227656242e-02, 4.949058493897810e-02, 0.000000000000000e+00, 4.917033014217492e-02, 4.516974112577109e-02, 0.000000000000000e+00, 5.496903928329294e-02, 4.516974112577109e-02, 0.000000000000000e+00, 5.496903928329294e-02, 2.864291974470999e+00, 0.000000000000000e+00, 2.554451778042131e+00, 2.760965859381902e+00, 0.000000000000000e+00, 2.423836960098544e+00, 3.085266208095684e-02, 0.000000000000000e+00, 2.793918018138355e-02, 6.301322453692907e+00, 0.000000000000000e+00, 6.047476873266916e+00, 2.515900352767639e+00, 0.000000000000000e+00, 1.629402416950184e+01, 2.515900352767635e+00, 0.000000000000000e+00, 1.629402416950184e+01, 7.041407357458702e+04, 0.000000000000000e+00, 5.868508094945669e+04, 6.119617168730468e+04, 0.000000000000000e+00, 4.977933318915997e+04, 3.559569082299518e+02, 0.000000000000000e+00, 3.022648227692841e+02, 3.241595634820426e+05, 0.000000000000000e+00, 3.409193213048052e+05, 9.918026780235268e+04, 0.000000000000000e+00, 5.382007608489258e+05, 9.918026780235274e+04, 0.000000000000000e+00, 5.382007608489258e+05, 9.936367956490476e-05, 0.000000000000000e+00, 9.943511840018150e-05, 9.930802442216480e-05, 0.000000000000000e+00, 9.938130782917756e-05, 9.936028930634903e-05, 0.000000000000000e+00, 9.943305582399345e-05, 9.931253693102303e-05, 0.000000000000000e+00, 9.938417381542180e-05, 9.933539712547092e-05, 0.000000000000000e+00, 9.940812528974266e-05, 9.933539712547092e-05, 0.000000000000000e+00, 9.940812528974266e-05, 1.871563109528728e-03, 0.000000000000000e+00, 1.871846856128181e-03, 1.830812056256027e-03, 0.000000000000000e+00, 1.832307587921146e-03, 1.906699178329619e-03, 0.000000000000000e+00, 1.896839855964448e-03, 1.870510291432628e-03, 0.000000000000000e+00, 1.860435649747737e-03, 1.823290775876607e-03, 0.000000000000000e+00, 1.847456725337902e-03, 1.823290775876607e-03, 0.000000000000000e+00, 1.847456725337902e-03, 6.918536915402185e-02, 0.000000000000000e+00, 6.973779529389867e-02, 5.421408901378604e-02, 0.000000000000000e+00, 5.414024991566970e-02, 9.269229285753529e-02, 0.000000000000000e+00, 8.317716461686046e-02, 8.207480742881676e-02, 0.000000000000000e+00, 7.440865455138823e-02, 6.149973175911057e-02, 0.000000000000000e+00, 7.169309896884339e-02, 6.149973175911090e-02, 0.000000000000000e+00, 7.169309896884325e-02, 1.870792881178634e+01, 0.000000000000000e+00, 1.818277273890588e+01, 2.708472151978478e+00, 0.000000000000000e+00, 2.668323110360484e+00, 2.527131360807196e+01, 0.000000000000000e+00, 2.159927502988043e+01, 1.981931078914540e-03, 0.000000000000000e+00, 1.984604577228130e-03, 1.248775698122979e+01, 0.000000000000000e+00, 1.050469270560129e+01, 1.248775698122979e+01, 0.000000000000000e+00, 1.050469270560129e+01, 7.677959556702846e+05, 0.000000000000000e+00, 6.840618960663988e+05, 3.639108823526908e+05, 0.000000000000000e+00, 3.478752948433860e+05, 8.703328825977490e+05, 0.000000000000000e+00, 7.327470545636452e+05, 7.726321710527429e+01, 0.000000000000000e+00, 7.585536103809524e+01, 8.168912543674281e+05, 0.000000000000000e+00, 3.153905796011646e+05, 8.168912543674273e+05, 0.000000000000000e+00, 3.153905796011647e+05, 6.884746298560138e-02, 0.000000000000000e+00, 6.805475036865131e-02, 7.005799965186876e-02, 0.000000000000000e+00, 6.924365152802968e-02, 6.963186991451614e-02, 0.000000000000000e+00, 6.882270778679626e-02, 6.927902987130445e-02, 0.000000000000000e+00, 6.848122847665251e-02, 6.945544567336198e-02, 0.000000000000000e+00, 6.865205678192750e-02, 6.945544567336198e-02, 0.000000000000000e+00, 6.865205678192750e-02, 7.468353301681430e-02, 0.000000000000000e+00, 7.394748126607084e-02, 1.182893317596118e-01, 0.000000000000000e+00, 1.167651550746390e-01, 1.030327730881584e-01, 0.000000000000000e+00, 1.016730838962172e-01, 9.024780821603808e-02, 0.000000000000000e+00, 8.930244077402924e-02, 9.645717093031422e-02, 0.000000000000000e+00, 9.539017376310824e-02, 9.645717093031422e-02, 0.000000000000000e+00, 9.539017376310824e-02, 4.713441612631317e-02, 0.000000000000000e+00, 4.690467279471430e-02, 1.344573764931554e+00, 0.000000000000000e+00, 1.321485975353676e+00, 7.812399093722693e-01, 0.000000000000000e+00, 7.581974449675797e-01, 3.367208965688956e-01, 0.000000000000000e+00, 3.311966635426017e-01, 5.074746400422829e-01, 0.000000000000000e+00, 5.080120773411618e-01, 5.074746400422845e-01, 0.000000000000000e+00, 5.080120773411617e-01, 1.521750790466154e-01, 0.000000000000000e+00, 1.490741930919076e-01, 3.767029635878047e+02, 0.000000000000000e+00, 3.690750935948902e+02, 1.547753666162319e+02, 0.000000000000000e+00, 1.402174554312666e+02, 3.813036689128671e-01, 0.000000000000000e+00, 3.645616589981365e-01, 4.036693861027452e+01, 0.000000000000000e+00, 3.428963543000756e+01, 4.036693861027448e+01, 0.000000000000000e+00, 3.428963543000756e+01, 2.405737105567678e+04, 0.000000000000000e+00, 2.161966678247349e+04, 1.868791008422476e+07, 0.000000000000000e+00, 1.856035610145812e+07, 2.218276496238957e+06, 0.000000000000000e+00, 1.843759643438376e+06, 4.589762829678638e+01, 0.000000000000000e+00, 4.371755664492991e+01, 9.612185642177264e+05, 0.000000000000000e+00, 4.095873450985811e+05, 9.612185642177287e+05, 0.000000000000000e+00, 4.095873450985824e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms2bs_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.230018804868608e+01, -2.230023812265617e+01, -2.230051090800583e+01, -2.229976552558109e+01, -2.230014893357673e+01, -2.230014893357673e+01, -3.399964600713623e+00, -3.399976303541089e+00, -3.400541076685387e+00, -3.402443219248886e+00, -3.401131335141577e+00, -3.401131335141577e+00, -6.665024493301204e-01, -6.662543869449268e-01, -6.615483829417689e-01, -6.664879353387061e-01, -6.678136628648084e-01, -6.678136628648084e-01, -2.029963751135302e-01, -2.047233327810735e-01, -7.712580882899804e-01, -1.647572366673568e-01, -1.968788960551433e-01, -1.968788960551432e-01, -9.109360731749543e-03, -9.591945293491310e-03, -5.192935107044543e-02, -5.254694448849960e-03, -7.332680884949601e-03, -7.332680884949601e-03, -5.409079967687360e+00, -5.409441094632442e+00, -5.409103917301339e+00, -5.409422578468729e+00, -5.409258988447435e+00, -5.409258988447435e+00, -2.119131048267925e+00, -2.134760234876077e+00, -2.116588478114362e+00, -2.130176201001532e+00, -2.129013530585411e+00, -2.129013530585411e+00, -5.966873943459052e-01, -6.264680645444287e-01, -5.374380109476093e-01, -5.351711711258167e-01, -6.068032857052156e-01, -6.068032857052157e-01, -1.245933062949372e-01, -2.126015609592257e-01, -1.164904591158764e-01, -1.812962169794820e+00, -1.399667463442703e-01, -1.399667463442703e-01, -4.057071344220179e-03, -5.136740095530260e-03, -3.933333632369829e-03, -8.143770772420110e-02, -4.940723682789978e-03, -4.940723682789979e-03, -6.038096665903443e-01, -6.030303679423650e-01, -6.033130820047545e-01, -6.035352830279288e-01, -6.034235320340050e-01, -6.034235320340050e-01, -5.841826064371027e-01, -5.259044707872733e-01, -5.426462745773054e-01, -5.590416177796368e-01, -5.504745555090347e-01, -5.504745555090347e-01, -6.475384358048474e-01, -2.587646892463830e-01, -2.982217796653011e-01, -3.607993209667612e-01, -3.295337299830299e-01, -3.295337299830300e-01, -4.776367888062221e-01, -4.975120275461149e-02, -6.692949362012526e-02, -3.434792143531313e-01, -1.003423993357981e-01, -1.003423993357982e-01, -1.283326773807986e-02, -1.373211132394278e-03, -2.887643179690405e-03, -9.504085402415247e-02, -4.537368591098761e-03, -4.537368591098758e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms2bs_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.829784652786687e+01, -2.829884947370052e+01, -2.829797381583470e+01, -2.829894178575377e+01, -2.829849922050356e+01, -2.829966142238206e+01, -2.829706890112235e+01, -2.829783682597644e+01, -2.829792045301520e+01, -2.829865241340999e+01, -2.829792045301520e+01, -2.829865241340999e+01, -4.169602063317904e+00, -4.169759760196394e+00, -4.169684731382711e+00, -4.169836441863976e+00, -4.171593020653926e+00, -4.172056606049254e+00, -4.169985378578761e+00, -4.170323183111498e+00, -4.168721467798692e+00, -4.172114355652277e+00, -4.168721467798692e+00, -4.172114355652277e+00, -7.821796722550832e-01, -7.868402265394837e-01, -7.804374616327067e-01, -7.861431351032713e-01, -7.577621821682188e-01, -7.500928705751589e-01, -7.586114450536088e-01, -7.609836357568989e-01, -7.952808446590917e-01, -7.129780766240744e-01, -7.952808446590917e-01, -7.129780766240744e-01, -2.071201411113446e-01, -2.222202651580463e-01, -2.068684512046750e-01, -2.241845943715464e-01, -8.969431519336791e-01, -9.354424909583339e-01, -1.632084423768446e-01, -1.692053675498087e-01, -2.136208612642058e-01, -1.368761834287023e-01, -2.136208612642057e-01, -1.368761834287022e-01, -1.172905399679421e-02, -1.246032457186969e-02, -1.228741520971535e-02, -1.315848588477064e-02, -6.498248715108512e-02, -6.821600137923696e-02, -7.060214536455566e-03, -6.942800853022484e-03, -1.046787153063029e-02, -5.961627227597853e-03, -1.046787153063029e-02, -5.961627227597853e-03, -6.997831060574049e+00, -6.996389946058076e+00, -6.999959462342134e+00, -6.998444895345958e+00, -6.998059275604049e+00, -6.996539841901464e+00, -6.999881064682223e+00, -6.998402384761540e+00, -6.998838279452482e+00, -6.997408323790303e+00, -6.998838279452482e+00, -6.997408323790303e+00, -2.381427508652985e+00, -2.386133922136091e+00, -2.423498798955231e+00, -2.426798127705212e+00, -2.385104708967166e+00, -2.388048605328587e+00, -2.423080111217081e+00, -2.426754752227029e+00, -2.404239656912655e+00, -2.407506362718097e+00, -2.404239656912655e+00, -2.407506362718097e+00, -7.281764549226734e-01, -7.276127567129939e-01, -8.530611341994484e-01, -8.552582453676573e-01, -6.387297116483830e-01, -6.706070099654463e-01, -6.899837347720349e-01, -7.271584630906185e-01, -7.710496601305714e-01, -7.232587134459293e-01, -7.710496601305714e-01, -7.232587134459295e-01, -1.389561573830102e-01, -1.391785930216084e-01, -2.212136389579193e-01, -2.216473130637460e-01, -1.293227376689048e-01, -1.339524679199917e-01, -2.446046614355689e+00, -2.445162296374886e+00, -1.480175712376774e-01, -1.497015676456841e-01, -1.480175712376774e-01, -1.497015676456842e-01, -5.297763813279459e-03, -5.505647524296438e-03, -6.793082839132296e-03, -6.895945766167643e-03, -5.078446343260263e-03, -5.377296642362599e-03, -9.807010345354916e-02, -9.897120969176561e-02, -5.188195191053781e-03, -7.120624122470886e-03, -5.188195191053782e-03, -7.120624122470887e-03, -8.001009742370464e-01, -8.033005356159200e-01, -7.887498715623020e-01, -7.920964963784329e-01, -7.925455392668043e-01, -7.958735351853525e-01, -7.958618935422588e-01, -7.990976170694727e-01, -7.941853364066268e-01, -7.974672177494504e-01, -7.941853364066268e-01, -7.974672177494504e-01, -7.841388108132116e-01, -7.864774893413018e-01, -6.334909761780536e-01, -6.366449625948825e-01, -6.678697094940761e-01, -6.714285163066962e-01, -7.075886289012465e-01, -7.103330613009445e-01, -6.857709194769952e-01, -6.890260860237222e-01, -6.857709194769952e-01, -6.890260860237221e-01, -8.808978178051875e-01, -8.873856553600251e-01, -2.743251859446746e-01, -2.754737369869447e-01, -3.127940807302846e-01, -3.158262390415804e-01, -3.974562627671518e-01, -3.996743808949602e-01, -3.513926219254240e-01, -3.512211677325108e-01, -3.513926219254241e-01, -3.512211677325109e-01, -5.397753573787029e-01, -5.479134073510531e-01, -6.403989873887836e-02, -6.443263942594153e-02, -8.299257542387112e-02, -8.522156629726051e-02, -3.904617928759600e-01, -4.049922240696706e-01, -1.137507465765084e-01, -1.157086316299207e-01, -1.137507465765083e-01, -1.157086316299207e-01, -1.674921801361228e-02, -1.735089333916034e-02, -1.828739601850876e-03, -1.832857634026477e-03, -3.720225121460046e-03, -3.956544996250880e-03, -1.097799451482016e-01, -1.116341604155369e-01, -4.913308292224564e-03, -6.527834078273248e-03, -4.913308292224560e-03, -6.527834078273242e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2bs_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.889528577500103e-09, 0.000000000000000e+00, -9.874837536267259e-09, -9.889289547270099e-09, 0.000000000000000e+00, -9.874665699703917e-09, -9.886542982416302e-09, 0.000000000000000e+00, -9.871286883837539e-09, -9.889148304899498e-09, 0.000000000000000e+00, -9.874631044779539e-09, -9.889429573851550e-09, 0.000000000000000e+00, -9.871847612164032e-09, -9.889429573851550e-09, 0.000000000000000e+00, -9.871847612164032e-09, -2.443612072401100e-05, 0.000000000000000e+00, -2.436932630049916e-05, -2.445213916485082e-05, 0.000000000000000e+00, -2.437448152049317e-05, -2.470279607872234e-05, 0.000000000000000e+00, -2.473143236848831e-05, -2.485492302133622e-05, 0.000000000000000e+00, -2.479919532379172e-05, -2.442182285944319e-05, 0.000000000000000e+00, -2.502867815954250e-05, -2.442182285944319e-05, 0.000000000000000e+00, -2.502867815954250e-05, -7.450350738776981e-03, 0.000000000000000e+00, -7.419205789253152e-03, -7.468268664458748e-03, 0.000000000000000e+00, -7.428212346788882e-03, -7.723170360132040e-03, 0.000000000000000e+00, -7.882281376557601e-03, -7.888717459695039e-03, 0.000000000000000e+00, -7.896037017663964e-03, -7.324930017375077e-03, 0.000000000000000e+00, -1.037660325076686e-02, -7.324930017375077e-03, 0.000000000000000e+00, -1.037660325076686e-02, -1.189640957706381e+00, 0.000000000000000e+00, -8.534535592188349e-01, -1.271454975212665e+00, 0.000000000000000e+00, -8.951373598034198e-01, -4.534198426679746e-03, 0.000000000000000e+00, -4.174887274201877e-03, -1.889377938711517e+00, 0.000000000000000e+00, -1.728100201373735e+00, -1.091468158635339e+00, 0.000000000000000e+00, -2.369241477022351e+00, -1.091468158635337e+00, 0.000000000000000e+00, -2.369241477022353e+00, -5.279688260410827e+00, 0.000000000000000e+00, -5.225389529165482e+00, -5.562232435939475e+00, 0.000000000000000e+00, -5.531861999504790e+00, -3.031418755411280e+00, 0.000000000000000e+00, -3.030427978440652e+00, -4.842478353969029e+00, 0.000000000000000e+00, -4.710881692400461e+00, -5.273464997737902e+00, 0.000000000000000e+00, -1.339358643336375e+01, -5.273464997737915e+00, 0.000000000000000e+00, -1.339358643336377e+01, -1.803042306112286e-06, 0.000000000000000e+00, -1.798932326963404e-06, -1.801799078359349e-06, 0.000000000000000e+00, -1.797716304095432e-06, -1.800650100862080e-06, 0.000000000000000e+00, -1.797214895978253e-06, -1.799667095045185e-06, 0.000000000000000e+00, -1.796192994930534e-06, -1.804167195225829e-06, 0.000000000000000e+00, -1.798615938470679e-06, -1.804167195225829e-06, 0.000000000000000e+00, -1.798615938470679e-06, -1.652919159510256e-04, 0.000000000000000e+00, -1.626313921268880e-04, -1.505696632870203e-04, 0.000000000000000e+00, -1.484254110102170e-04, -1.498189656430105e-04, 0.000000000000000e+00, -1.519883585740349e-04, -1.364687258547576e-04, 0.000000000000000e+00, -1.381595620534500e-04, -1.678301655392991e-04, 0.000000000000000e+00, -1.565827279403227e-04, -1.678301655392991e-04, 0.000000000000000e+00, -1.565827279403227e-04, -5.084429730701856e-02, 0.000000000000000e+00, -5.095581271619123e-02, -5.094756776279210e-02, 0.000000000000000e+00, -5.133647908209475e-02, -5.124245944082355e-02, 0.000000000000000e+00, -5.904265631126502e-02, -3.172516782537491e-02, 0.000000000000000e+00, -4.213067339542156e-02, -4.296489598457032e-02, 0.000000000000000e+00, -5.574297434423629e-02, -4.296489598457032e-02, 0.000000000000000e+00, -5.574297434423629e-02, -2.003438544350745e+00, 0.000000000000000e+00, -2.055228944988890e+00, -6.651420312129982e-01, 0.000000000000000e+00, -6.675626377104145e-01, -2.308090154844352e+00, 0.000000000000000e+00, -2.199669328566677e+00, -3.144555244142944e-04, 0.000000000000000e+00, -3.153337875482596e-04, -2.088519515315262e+00, 0.000000000000000e+00, -2.371352929089805e+00, -2.088519515315261e+00, 0.000000000000000e+00, -2.371352929089803e+00, -6.798469295949563e+00, 0.000000000000000e+00, -5.884441700622995e+00, -5.854521104960086e+00, 0.000000000000000e+00, -5.405367122435872e+00, -3.334004485310625e+01, 0.000000000000000e+00, -3.708727422419501e+01, -3.470406864716132e+00, 0.000000000000000e+00, -3.256585114544876e+00, -1.657916813589240e+01, 0.000000000000000e+00, -1.627505044591523e+01, -1.657916813589235e+01, 0.000000000000000e+00, -1.627505044591521e+01, -2.919817079936349e-02, 0.000000000000000e+00, -2.834393593516287e-02, -2.631037422932762e-02, 0.000000000000000e+00, -2.560770259401734e-02, -2.725330956681717e-02, 0.000000000000000e+00, -2.650144602352321e-02, -2.809310047067103e-02, 0.000000000000000e+00, -2.730177280152274e-02, -2.767643444287469e-02, 0.000000000000000e+00, -2.690134309441328e-02, -2.767643444287469e-02, 0.000000000000000e+00, -2.690134309441328e-02, -5.224307934044289e-02, 0.000000000000000e+00, -5.040066430479667e-02, -3.031377623561012e-02, 0.000000000000000e+00, -2.983517469758360e-02, -3.442308212146109e-02, 0.000000000000000e+00, -3.383044363634799e-02, -4.040537556560922e-02, 0.000000000000000e+00, -3.981005807143903e-02, -3.865662724147958e-02, 0.000000000000000e+00, -3.744836158056121e-02, -3.865662724147960e-02, 0.000000000000000e+00, -3.744836158056126e-02, -3.743940408722653e-02, 0.000000000000000e+00, -3.916252002420859e-02, -3.491847886091699e-01, 0.000000000000000e+00, -3.478855592900892e-01, -3.153582983976691e-01, 0.000000000000000e+00, -3.120914486006675e-01, -2.846727055646325e-01, 0.000000000000000e+00, -2.800242769481903e-01, -3.066512014306687e-01, 0.000000000000000e+00, -3.096881238256849e-01, -3.066512014306688e-01, 0.000000000000000e+00, -3.096881238256850e-01, -9.506792888112628e-02, 0.000000000000000e+00, -9.006706527940508e-02, -2.740981347912871e+00, 0.000000000000000e+00, -2.749128679321561e+00, -2.709790276296259e+00, 0.000000000000000e+00, -2.758445771983460e+00, -3.931014430707830e-01, 0.000000000000000e+00, -3.769548660300874e-01, -3.311600889910828e+00, 0.000000000000000e+00, -3.817082253341531e+00, -3.311600889910828e+00, 0.000000000000000e+00, -3.817082253341539e+00, -4.188846993353932e+00, 0.000000000000000e+00, -4.276036096039359e+00, -2.092023022805749e+01, 0.000000000000000e+00, -3.705647045488706e+01, -1.290982941669176e+01, 0.000000000000000e+00, -1.374299874242530e+01, -3.643534060230125e+00, 0.000000000000000e+00, -3.360604632402347e+00, -3.417981965454967e+01, 0.000000000000000e+00, -1.689795496602078e+01, -3.417981965454973e+01, 0.000000000000000e+00, -1.689795496602083e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2bs_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2bs_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.032908044981573e-04, 3.022068824909700e-04, 3.032804618945430e-04, 3.021994933414971e-04, 3.031048605687453e-04, 3.019878166213164e-04, 3.032146722359873e-04, 3.021297211203489e-04, 3.032876755434597e-04, 3.019689833796795e-04, 3.032876755434597e-04, 3.019689833796795e-04, 3.007277994656311e-03, 2.989049181975057e-03, 3.011353233152627e-03, 2.990242793240243e-03, 3.073990906243123e-03, 3.081026868000791e-03, 3.119944585051108e-03, 3.104953446470953e-03, 3.000247353987825e-03, 3.163138546111765e-03, 3.000247353987825e-03, 3.163138546111765e-03, 1.076660485736391e-05, 5.063846498696388e-06, 1.468644612205646e-05, 5.090869635746936e-06, 9.409244591584870e-05, 2.510064185173083e-04, 5.709393278664794e-04, 5.927474248302820e-04, 4.452804170777373e-06, 3.937531851770628e-03, 4.452804170777373e-06, 3.937531851770628e-03, 1.545603343899223e-02, 7.239185366593440e-03, 1.884492471347594e-02, 1.012428291236033e-02, 2.213329608661815e-04, 2.667933464993661e-04, 7.906994129346152e-03, 8.485469312914242e-03, 1.752250873097801e-02, 8.989586170307222e-04, 1.752250873097796e-02, 8.989586170307205e-04, 8.411580489401770e-10, 4.848458458080738e-10, 1.884175416643771e-09, 1.503526626196955e-09, 1.872362581905756e-05, 2.240354799321507e-05, 4.368922219452425e-10, 3.489316650538393e-10, 1.373068747508921e-09, 9.799577957137870e-10, 1.373068747508964e-09, 9.799577957137866e-10, 1.129458165672121e-04, 1.065003723076767e-04, 1.120430214965927e-04, 1.056084140963585e-04, 1.103146990156302e-04, 1.046090157721457e-04, 1.096320956910831e-04, 1.038900287634730e-04, 1.144406260123565e-04, 1.063791557717843e-04, 1.144406260123565e-04, 1.063791557717843e-04, 5.184921502904345e-03, 5.059004716831362e-03, 4.655137942779336e-03, 4.541481694843014e-03, 4.320439458558360e-03, 4.458180450684504e-03, 3.805980718184297e-03, 3.923619019184608e-03, 5.520822059934671e-03, 4.875874098037410e-03, 5.520822059934671e-03, 4.875874098037410e-03, 5.161704907389804e-02, 5.175408242821279e-02, 6.623931977568311e-02, 6.774045998198139e-02, 2.637441819388624e-02, 4.269039959142078e-02, 1.063366238900975e-02, 2.562838270278596e-02, 5.111608470807038e-02, 5.418223118032826e-02, 5.111608470807035e-02, 5.418223118032824e-02, 1.598805535359623e-03, 1.757509348995436e-03, 6.149855990103346e-03, 6.520858772701353e-03, 1.204420906553651e-03, 1.448386880245811e-03, 6.644630644588541e-03, 6.659933864411025e-03, 3.206633026603237e-03, 4.914965277665450e-03, 3.206633026603236e-03, 4.914965277665439e-03, 4.180750500302758e-11, 5.938488138858389e-11, 5.324533321390351e-10, 4.795452994295629e-10, 1.547997888721176e-09, 2.482899638442571e-09, 3.401771435156525e-04, 2.932812625034188e-04, 5.849060641411358e-12, 7.934284920408846e-10, 5.849060641411675e-12, 7.934284920408731e-10, 2.817215483944990e-02, 2.744952416079535e-02, 2.328629242587036e-02, 2.274963955129895e-02, 2.487936102804082e-02, 2.428397998445574e-02, 2.630005234307021e-02, 2.565498728389920e-02, 2.559270723539951e-02, 2.496739413246777e-02, 2.559270723539951e-02, 2.496739413246778e-02, 5.464315299455524e-02, 5.301188549487835e-02, 1.047618081343108e-02, 1.042947076809195e-02, 1.827064668886749e-02, 1.817280776323574e-02, 2.968354476733077e-02, 2.954054311282554e-02, 2.494592875670098e-02, 2.418226311059012e-02, 2.494592875670100e-02, 2.418226311059017e-02, 5.045023521297695e-02, 5.497110347915799e-02, 6.555220086360358e-03, 6.846872721798763e-03, 1.899810795243274e-02, 1.965589805675590e-02, 5.137138823558424e-02, 5.112177529825461e-02, 3.578865301055246e-02, 3.619395166747677e-02, 3.578865301055244e-02, 3.619395166747678e-02, 4.201421633247135e-02, 4.081859033461908e-02, 7.529141560993108e-06, 9.283492218605117e-06, 6.842431143443077e-05, 8.167808142626898e-05, 6.218090470690145e-02, 7.048015662958564e-02, 9.054614312765608e-04, 1.465698232232672e-03, 9.054614312765546e-04, 1.465698232232688e-03, 2.872088193485011e-08, 3.344929754504121e-08, 2.489147384293703e-14, 1.172280633201524e-13, 4.618818372970360e-11, 7.134521069788818e-11, 7.886542381514274e-04, 4.681822382036835e-04, 1.600155249112315e-10, 5.558852735574936e-10, 1.600155249112305e-10, 5.558852735575076e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05

import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_xpbe_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.491246130354375e-02, -7.491308804011224e-02, -7.491492351190622e-02, -7.490561916699925e-02, -7.491062938217094e-02, -7.491062938217094e-02, -5.081042169567564e-02, -5.081434775378624e-02, -5.090548103390248e-02, -5.074895317168623e-02, -5.081534179617444e-02, -5.081534179617444e-02, -2.852397954271509e-02, -2.831887102347584e-02, -2.335522368765976e-02, -2.362027578240770e-02, -2.362973626877331e-02, -2.362973626877331e-02, -5.564147489085659e-03, -6.087777874679706e-03, -3.140742413197133e-02, -1.666170916536779e-03, -1.847537011690686e-03, -1.847537011690686e-03, -3.072645741140911e-09, -4.106580308443908e-09, -5.288671221726632e-06, -2.176409161398773e-10, -5.652023024951824e-10, -5.652023024951824e-10, -6.695657692736022e-02, -6.711139739398611e-02, -6.696296934684885e-02, -6.709964277888943e-02, -6.703533648396784e-02, -6.703533648396784e-02, -2.928944875326946e-02, -2.976121461871196e-02, -2.830688959074067e-02, -2.871815159807858e-02, -3.003667807399719e-02, -3.003667807399719e-02, -3.780884659208560e-02, -5.421092121178386e-02, -3.518572502699180e-02, -4.947240000815525e-02, -3.945720476121652e-02, -3.945720476121649e-02, -3.008949993689482e-04, -2.851703297348865e-03, -2.289078495324587e-04, -7.064326397509815e-02, -8.278230036552134e-04, -8.278230036552134e-04, -8.454532451733499e-11, -2.303418610363761e-10, -4.075443949327906e-10, -5.422228181457822e-05, -4.010034817224228e-10, -4.010034834571463e-10, -5.952091248775931e-02, -5.334510894765497e-02, -5.532340803221122e-02, -5.710871357825547e-02, -5.619784188635276e-02, -5.619784188635276e-02, -6.112987841878293e-02, -2.698043401169321e-02, -3.365795859101682e-02, -4.210410730894758e-02, -3.761003585741984e-02, -3.761003585741984e-02, -5.405624516205663e-02, -5.381470082363754e-03, -9.617379201069483e-03, -2.255700425552703e-02, -1.532513920845179e-02, -1.532513920845179e-02, -2.604478917823407e-02, -3.975719245451204e-06, -1.596317784820223e-05, -2.694610589453330e-02, -1.873821601328096e-04, -1.873821601328131e-04, -1.214152427382598e-08, -2.123218051028330e-12, -3.553731332643639e-11, -1.412916211916034e-04, -3.603352030819018e-10, -3.603352011303379e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_xpbe_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.320663000257415e-01, -1.320664875979734e-01, -1.320668515023536e-01, -1.320671678006808e-01, -1.320689934208151e-01, -1.320686437208014e-01, -1.320591211213812e-01, -1.320602305763888e-01, -1.320636700614402e-01, -1.320655265338400e-01, -1.320636700614402e-01, -1.320655265338400e-01, -1.048512652362314e-01, -1.048528299161438e-01, -1.048548434730763e-01, -1.048566302922650e-01, -1.049426032966427e-01, -1.049402386242226e-01, -1.047957442073347e-01, -1.047941364785451e-01, -1.048782247753275e-01, -1.048353205924002e-01, -1.048782247753275e-01, -1.048353205924002e-01, -7.368351602360841e-02, -7.337363364908442e-02, -7.349583480935461e-02, -7.311721085015081e-02, -6.699216018198169e-02, -6.746031451801238e-02, -6.770085760569179e-02, -6.755311270272149e-02, -6.549897309055912e-02, -6.996996937878727e-02, -6.549897309055912e-02, -6.996996937878727e-02, -2.538545010705277e-02, -2.452656510608473e-02, -2.724202049464185e-02, -2.618861255584970e-02, -7.905812824778070e-02, -7.628719212686789e-02, -9.075967065706476e-03, -8.970367853217850e-03, -8.803252958366221e-03, -1.513619067343417e-02, -8.803252958366247e-03, -1.513619067343417e-02, -2.050950611349947e-08, -1.973829398765470e-08, -2.748572958092099e-08, -2.631161901693978e-08, -3.465096865148183e-05, -3.322453540384733e-05, -1.421098951594540e-09, -1.435494433362971e-09, -3.494822380328919e-09, -4.789774259562982e-09, -3.494822378160515e-09, -4.789774256960897e-09, -1.210470213207066e-01, -1.210812603621561e-01, -1.211803157915353e-01, -1.212155306455689e-01, -1.210522368186574e-01, -1.210871160335686e-01, -1.211705719863441e-01, -1.212049891001734e-01, -1.211147681326980e-01, -1.211496713115335e-01, -1.211147681326980e-01, -1.211496713115335e-01, -7.970664791370043e-02, -7.971046883755882e-02, -8.033532654943534e-02, -8.035616202515899e-02, -7.844440532552482e-02, -7.831723863969924e-02, -7.901940546561427e-02, -7.888545147889424e-02, -8.053165370269960e-02, -8.087016574262974e-02, -8.053165370269960e-02, -8.087016574262974e-02, -7.893862445435111e-02, -7.919134205887438e-02, -8.281060952298906e-02, -8.275599606700432e-02, -7.820285079041979e-02, -7.493173942959780e-02, -8.204004844731314e-02, -7.835314035850512e-02, -7.763099783243378e-02, -8.264998917746620e-02, -7.763099783243378e-02, -8.264998917746617e-02, -1.819716702828864e-03, -1.805471693318122e-03, -1.458318535157920e-02, -1.452062855688702e-02, -1.423553866293619e-03, -1.363437927569931e-03, -1.122863670758814e-01, -1.123548929608557e-01, -4.865668389674549e-03, -4.635549323885958e-03, -4.865668389674549e-03, -4.635549323885958e-03, -5.622435421929728e-10, -5.496771944909751e-10, -1.518598740108616e-09, -1.505003069946019e-09, -2.727276694290292e-09, -2.637190134116868e-09, -3.407772523396881e-04, -3.390991589330847e-04, -2.993556279761150e-09, -2.491442002087573e-09, -2.993556283040861e-09, -2.491442003198880e-09, -7.690886895200895e-02, -7.642706525637730e-02, -8.051265616040758e-02, -8.005219831652097e-02, -7.968419352411309e-02, -7.921479362506156e-02, -7.868035058906991e-02, -7.820764895285461e-02, -7.922377227189292e-02, -7.875276060038279e-02, -7.922377227189292e-02, -7.875276060038279e-02, -7.395924360107516e-02, -7.354450670781375e-02, -6.943337646007000e-02, -6.910760623216536e-02, -7.511734029294405e-02, -7.472864486132705e-02, -7.914511777235120e-02, -7.878760677604355e-02, -7.739615344274674e-02, -7.704545978667676e-02, -7.739615344274674e-02, -7.704545978667676e-02, -8.442870656065851e-02, -8.423218050468695e-02, -2.471635704690859e-02, -2.459016998425879e-02, -3.802183297466256e-02, -3.767091070514667e-02, -6.164926278696672e-02, -6.128454048752988e-02, -5.055571507537961e-02, -5.057341399085419e-02, -5.055571507537961e-02, -5.057341399085421e-02, -6.781974735102500e-02, -6.731565003012206e-02, -2.557402867969350e-05, -2.544041826951165e-05, -1.028646974775533e-04, -1.002212253927479e-04, -6.529863333451676e-02, -6.418555371174552e-02, -1.172247231954185e-03, -1.121131854693423e-03, -1.172247231954198e-03, -1.121131854693458e-03, -8.015531000550612e-08, -7.830668922474725e-08, -1.403387902181397e-11, -1.401627905952779e-11, -2.384742792701814e-10, -2.302013856153332e-10, -8.754448407194649e-04, -8.639262987031595e-04, -2.647273946427653e-09, -2.245823751509282e-09, -2.647273944191486e-09, -2.245823750140477e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_xpbe_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.360789190209202e-10, 2.721578380418404e-10, 1.360789190209202e-10, 1.360810253788056e-10, 2.721620507576112e-10, 1.360810253788056e-10, 1.360833912713935e-10, 2.721667825427869e-10, 1.360833912713935e-10, 1.360522044679352e-10, 2.721044089358704e-10, 1.360522044679352e-10, 1.360695379370319e-10, 2.721390758740638e-10, 1.360695379370319e-10, 1.360695379370319e-10, 2.721390758740638e-10, 1.360695379370319e-10, 8.164515202138389e-07, 1.632903040427678e-06, 8.164515202138389e-07, 8.166043536555970e-07, 1.633208707311194e-06, 8.166043536555970e-07, 8.200519459036324e-07, 1.640103891807264e-06, 8.200519459036324e-07, 8.133115689793081e-07, 1.626623137958616e-06, 8.133115689793081e-07, 8.165374811464077e-07, 1.633074962292815e-06, 8.165374811464077e-07, 8.165374811464077e-07, 1.633074962292815e-06, 8.165374811464077e-07, 1.527152424031272e-03, 3.054304848062545e-03, 1.527152424031272e-03, 1.516634300541518e-03, 3.033268601083036e-03, 1.516634300541518e-03, 1.259387779562818e-03, 2.518775559125636e-03, 1.259387779562818e-03, 1.228711301748889e-03, 2.457422603497778e-03, 1.228711301748889e-03, 1.246049082954302e-03, 2.492098165908603e-03, 1.246049082954302e-03, 1.246049082954302e-03, 2.492098165908603e-03, 1.246049082954302e-03, 1.395870539113704e-01, 2.791741078227408e-01, 1.395870539113704e-01, 1.473911695746976e-01, 2.947823391493953e-01, 1.473911695746976e-01, 8.143444972535711e-04, 1.628688994507142e-03, 8.143444972535711e-04, 9.748550878444245e-02, 1.949710175688847e-01, 9.748550878444245e-02, 8.271675141349029e-02, 1.654335028269806e-01, 8.271675141349029e-02, 8.271675141349032e-02, 1.654335028269806e-01, 8.271675141349032e-02, 2.420368849334965e-03, 4.840737698371210e-03, 2.420368849334965e-03, 2.851771384303125e-03, 5.703542768625387e-03, 2.851771384303125e-03, 1.671885603346796e-02, 3.343771206694182e-02, 1.671885603346796e-02, 8.478853923976646e-04, 1.695770786223496e-03, 8.478853923976646e-04, 1.251185293953487e-03, 2.502370587805732e-03, 1.251185293953487e-03, 1.251185295390143e-03, 2.502370590844113e-03, 1.251185295390143e-03, 2.053296895263878e-07, 4.106593790527756e-07, 2.053296895263878e-07, 2.066970184763649e-07, 4.133940369527297e-07, 2.066970184763649e-07, 2.053842270231643e-07, 4.107684540463285e-07, 2.053842270231643e-07, 2.065911437082947e-07, 4.131822874165894e-07, 2.065911437082947e-07, 2.060249441153818e-07, 4.120498882307635e-07, 2.060249441153818e-07, 2.060249441153818e-07, 4.120498882307635e-07, 2.060249441153818e-07, 5.564661951861601e-06, 1.112932390372320e-05, 5.564661951861601e-06, 5.510499623961910e-06, 1.102099924792382e-05, 5.510499623961910e-06, 5.386156877005060e-06, 1.077231375401012e-05, 5.386156877005060e-06, 5.339620964580646e-06, 1.067924192916129e-05, 5.339620964580646e-06, 5.624047227100524e-06, 1.124809445420105e-05, 5.624047227100524e-06, 5.624047227100524e-06, 1.124809445420105e-05, 5.624047227100524e-06, 5.208684735686782e-03, 1.041736947137356e-02, 5.208684735686782e-03, 7.839449463110050e-03, 1.567889892622010e-02, 7.839449463110050e-03, 6.764159798740084e-03, 1.352831959748017e-02, 6.764159798740084e-03, 1.164271947755477e-02, 2.328543895510954e-02, 1.164271947755477e-02, 5.252517575592321e-03, 1.050503515118464e-02, 5.252517575592321e-03, 5.252517575592321e-03, 1.050503515118464e-02, 5.252517575592321e-03, 4.982995972480019e-02, 9.965991944960018e-02, 4.982995972480019e-02, 5.182354088007274e-02, 1.036470817601455e-01, 5.182354088007274e-02, 4.958411902217238e-02, 9.916823804434427e-02, 4.958411902217238e-02, 5.312768039800424e-05, 1.062553607960085e-04, 5.312768039800424e-05, 9.929704990791240e-02, 1.985940998158249e-01, 9.929704990791240e-02, 9.929704990791240e-02, 1.985940998158249e-01, 9.929704990791240e-02, 8.230468436547111e-04, 1.646093687237448e-03, 8.230468436547111e-04, 1.042027515651840e-03, 2.084055033627310e-03, 1.042027515651840e-03, 1.033559368509085e-02, 2.067118736879819e-02, 1.033559368509085e-02, 4.441235049631492e-02, 8.882470099263083e-02, 4.441235049631492e-02, 3.968069349823852e-03, 7.936138702552916e-03, 3.968069349823852e-03, 3.968069357491972e-03, 7.936138720240321e-03, 3.968069357491972e-03, 1.525018344568000e-02, 3.050036689136001e-02, 1.525018344568000e-02, 1.158445069421394e-02, 2.316890138842787e-02, 1.158445069421394e-02, 1.264298335468361e-02, 2.528596670936721e-02, 1.264298335468361e-02, 1.368859214829034e-02, 2.737718429658067e-02, 1.368859214829034e-02, 1.314402614044127e-02, 2.628805228088255e-02, 1.314402614044127e-02, 1.314402614044127e-02, 2.628805228088255e-02, 1.314402614044127e-02, 1.901743228105075e-02, 3.803486456210151e-02, 1.901743228105075e-02, 6.919359269512191e-03, 1.383871853902438e-02, 6.919359269512191e-03, 8.101516842553631e-03, 1.620303368510726e-02, 8.101516842553631e-03, 1.016959118359876e-02, 2.033918236719753e-02, 1.016959118359876e-02, 8.997147930341060e-03, 1.799429586068212e-02, 8.997147930341060e-03, 8.997147930341060e-03, 1.799429586068212e-02, 8.997147930341060e-03, 6.088955746866003e-03, 1.217791149373201e-02, 6.088955746866003e-03, 3.953984861799293e-02, 7.907969723598589e-02, 3.953984861799293e-02, 3.601734809723307e-02, 7.203469619446615e-02, 3.601734809723307e-02, 3.103611283135297e-02, 6.207222566270593e-02, 3.103611283135297e-02, 3.511007923935327e-02, 7.022015847870654e-02, 3.511007923935327e-02, 3.511007923935330e-02, 7.022015847870662e-02, 3.511007923935330e-02, 9.997903471116882e-03, 1.999580694223377e-02, 9.997903471116882e-03, 1.365009204895310e-02, 2.730018409790937e-02, 1.365009204895310e-02, 2.196705634947205e-02, 4.393411269894336e-02, 2.196705634947205e-02, 5.012818690098326e-02, 1.002563738019665e-01, 5.012818690098326e-02, 8.183863590599166e-02, 1.636772718119820e-01, 8.183863590599166e-02, 8.183863590599461e-02, 1.636772718119903e-01, 8.183863590599461e-02, 3.056515593062892e-03, 6.113031186227774e-03, 3.056515593062892e-03, 1.113772003517426e-03, 2.227543998094577e-03, 1.113772003517426e-03, 1.402393458730415e-03, 2.804786923667602e-03, 1.402393458730415e-03, 7.402296621065627e-02, 1.480459324213129e-01, 7.402296621065627e-02, 5.058028751713884e-03, 1.011605750458217e-02, 5.058028751713884e-03, 5.058028760021664e-03, 1.011605752224440e-02, 5.058028760021664e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
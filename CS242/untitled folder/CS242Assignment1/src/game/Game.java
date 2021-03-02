package game;
import unocard.*;

public class Game {
	
	private final static int NUM_HAND_CARDS = 7;
	
	private SpecialEffectsManager em;
	private PlayersManager pm;
	
	public Game() {
		this.pm = new PlayersManager();
		this.em = new SpecialEffectsManager(this.pm);
	}
	
	
	public void prepareGame() {
		this.em.drawPile.shuffle();
		UnoCard card = this.em.drawPile.getCard(0);
		
		while(card.getColor().equals("BLACK")){
			this.em.drawPile.addCard(card);
			card = this.em.drawPile.getCard(0);
		}
		
		this.em.discardPile.addCard(card);
		
		//////
		
		Player currPlayer = this.pm.getCurrentPlayer();
		int startID = currPlayer.getPid();	
		
		for(int i = 0; i < NUM_HAND_CARDS; i++){
			card = this.em.drawCardWithRefill();
			currPlayer.addCard(card);
		}
			this.pm.rotatePlayer();
			currPlayer = this.pm.getCurrentPlayer();
		
		while (currPlayer.getPid() != startID) {
			
			for(int i = 0; i < NUM_HAND_CARDS; i++){
				card = this.em.drawCardWithRefill();
				currPlayer.addCard(card);
			}
				this.pm.rotatePlayer();
				currPlayer = this.pm.getCurrentPlayer();
		}
	}
	
	/**
	 * The current player draw card from the deck.
	 */
	public void playerDrawCard(){
		Player currPlayer = this.pm.getCurrentPlayer();
		UnoCard card = this.em.drawCardWithRefill();
		currPlayer.addCard(card);
	}
	
	/**
	 * The current player plays card from hand. If can't player take back the card.
	 * @return true if successfully played. Else returns false.
	 */
	public boolean playerPlayCard(String name){
		Player currPlayer = this.pm.getCurrentPlayer();
		UnoCard card = currPlayer.playCard(name);
		
		if (this.em.tryPlayCard(card)) {
			return true;
		} else {
			currPlayer.addCard(card);
			return false;
		}
	}	
	
	/**
	 * apply the card effect on top of discard pile
	 */
	public void doEffectInGame() {
		UnoCard card = this.em.discardPile.getNextCard();
		card.doSpecialEffect(em);
	}
	
	/**
	 * Player declares the next color to be matched (may be used on any 
	 * turn even if the player has matching color; current color may be 
	 * chosen as the next to be matched.
	 * Return true if successful. False otherwise.
	 */
	public boolean setWild(String wildColor){
		this.em.setWildColor(wildColor);
		if(this.em.wildCardColor == null) {
			return false;
		}
		this.doEffectInGame();
		return true;
	}
	
	/**
	 * End the current player's turn.
	 * @param saysUno: true if the current player says uno first.
	 */
	public void finishPlayerTurn(boolean saysUno) {
		
		Player currPlayer = this.pm.getCurrentPlayer();
		
		if(saysUno && currPlayer.getNumCards() != 1){
			UnoCard card1 = this.em.drawCardWithRefill();
			UnoCard card2 = this.em.drawCardWithRefill();
			currPlayer.addCard(card1);
			currPlayer.addCard(card2);
		} 
		
		if((!saysUno) && currPlayer.getNumCards() == 1) {
			UnoCard card1 = this.em.drawCardWithRefill();
			UnoCard card2 = this.em.drawCardWithRefill();
			currPlayer.addCard(card1);
			currPlayer.addCard(card2);
		}
		
		this.pm.rotatePlayer();
	}
	
	/**
	 * Game ends when a player has zero cards in hand.
	 */
	public void finishGame() {
		int winner_idx = this.pm.findWinner();
		if(winner_idx == -1){
			System.out.println("\n No winner yet.");
		}
		else {
			Player winner = this.pm.allPlayers.get(winner_idx);
			System.out.println("\n The winner is " + winner.getPname() + ".");
		}
		
	}
	
	
	
}

package game;
import java.util.ArrayList;
import unocard.*;

public class SpecialEffectsManager {
	
	PlayersManager pm;
	public DrawPile drawPile;
	public CardPile discardPile;
	public String wildCardColor;
	
	public SpecialEffectsManager (PlayersManager pm){
		this.pm = pm;
		this.drawPile = new DrawPile();
		this.discardPile = new CardPile();
	}
	
	/**
	 * Draw a Uno Card from top of the draw pile.
	 * Reusing discard pile when no card is left in the draw pile 
	 * @return the card on top of the draw pile.
	 */
	public UnoCard drawCardWithRefill() {
		int numOfCards = this.drawPile.getNumUnoCards();
		
		if(numOfCards == 0){
			while(this.discardPile.getNumUnoCards() > 1){
				this.drawPile.addCard(this.discardPile.removeCard(0));
			}
			this.drawPile.shuffle();
		}	
			
		return this.drawPile.getCard(0);
	}
	
	
	/**
	 * Check whether a card is playable.
	 * If true, play it and put it in discard pile.
	 * @param cardName the name of the card that will be played.
	 * @return true if the operation was successful.
	 */
	public boolean tryPlayCard(UnoCard card){		
		if(card == null) {
			return false;
		}
		if(this.drawPile.getNextCard().doMatch(card)){
			this.discardPile.addCard(card);
			return true;
		} else{
			return false;
		}		
	}
	

	/**
	 * Next player in sequence misses a turn.
	 */
	public void doSkip(){
		this.pm.rotatePlayer();
	}
	
	/**
	 * Order of play switches directions (clockwise to counterclockwise, or vice versa)
	 */
	public void doReverse(){
		this.pm.reversePlayerDirection();
	}
	
	/**
	 * Next player in sequence draws two cards and misses a turn, unless they have 
	 * another Draw Two card to "stack" the number of cards to draw for the next player
	 */
	public void doDrawTwo() {
		UnoCard firstCard = drawCardWithRefill();
		UnoCard secondCard = drawCardWithRefill();
		this.pm.getNextPlayer().addCard(firstCard);
		this.pm.getNextPlayer().addCard(secondCard);
		
		this.pm.rotatePlayer();
	}
	
	
	public void setWildColor(String wildColor){
		for (NumberCard.Color cardColor: NumberCard.Color.values()) {
			if(cardColor.name().equals(wildColor.toUpperCase())) {
				this.wildCardColor = wildColor;
			} else {
				this.wildCardColor = null;
			}
		}
	}
	
	
	/**
	 * Player declares the next color to be matched (may be used on any 
	 * turn even if the player has matching color; current color may be 
	 * chosen as the next to be matched.
	 */
	public String doWild() {
		return this.wildCardColor;
	}
	
	/**
	 * Player declares the next color to be matched; the next player in sequence 
	 * draws four cards and misses a turn unless they have another Draw Four Wild 
	 * card to "stack" the number of cards to draw for the next player 
	 */
	public void doDrawFourWild() {
		
		UnoCard firstCard = drawCardWithRefill();
		UnoCard secondCard = drawCardWithRefill();
		UnoCard thirdCard = drawCardWithRefill();
		UnoCard fourthCard = drawCardWithRefill();
		
		this.pm.getNextPlayer().addCard(firstCard);
		this.pm.getNextPlayer().addCard(secondCard);
		this.pm.getNextPlayer().addCard(thirdCard);
		this.pm.getNextPlayer().addCard(fourthCard);
		
		this.pm.rotatePlayer();
		
	}
	
	
}
